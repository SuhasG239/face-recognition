"""Face detection, face embeddings, and the database of registered people.

Uses two free pretrained models from the OpenCV model zoo (downloaded on first use):
- YuNet finds faces and five landmarks (eyes, nose, mouth corners).
- SFace turns an aligned face into 128 numbers (an "embedding"). Photos of the same
  person give similar embeddings, so recognizing someone means finding the closest
  registered embedding. No retraining is needed to add people.
"""
import json
import os
import urllib.request

import cv2
import numpy as np

# OpenCV 5 warns about every model it loads; only show real errors
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, 'models')
DB_DIR = os.path.join(HERE, 'faces_db')
DB_FILE = os.path.join(DB_DIR, 'embeddings.npz')
CONFIG_FILE = os.path.join(HERE, 'recognition_config.json')

MODEL_URLS = {
    'face_detection_yunet_2023mar.onnx':
        'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    'face_recognition_sface_2021dec.onnx':
        'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
}

# Used until the notebook writes recognition_config.json (SFace's published cosine threshold)
DEFAULT_THRESHOLD = 0.363
MAX_SIDE = 1280   # large photos are shrunk to this before detection


def ensure_models():
    """Download the two model files if they aren't in models/ yet. Returns their paths."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = []
    for name, url in MODEL_URLS.items():
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            print(f'Downloading {name}...')
            urllib.request.urlretrieve(url, path)
        paths.append(path)
    return paths


def load_threshold():
    """Similarity a face needs to count as a registered person (chosen in the notebook)."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)['threshold']
    return DEFAULT_THRESHOLD


def shrink(image, max_side=MAX_SIDE):
    """Scale an image down so its longer side is at most max_side."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class FaceEngine:
    """Finds faces in BGR images and computes their embeddings."""

    def __init__(self, min_confidence=0.8):
        detector_path, recognizer_path = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(detector_path, '', (320, 320), min_confidence, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, '')

    def detect(self, image):
        """Faces as an (n, 15) array: x, y, w, h, 10 landmark coordinates, confidence."""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return np.empty((0, 15), np.float32) if faces is None else faces

    def align(self, image, face):
        """The face cropped and rotated to the 112x112 layout SFace expects."""
        return self.recognizer.alignCrop(image, face)

    def embed(self, image, face):
        """128-number embedding of one detected face, scaled to length 1."""
        feature = self.recognizer.feature(self.align(image, face)).flatten()
        return feature / np.linalg.norm(feature)

    def analyze(self, image):
        """Every face in the image as (box, confidence, embedding). box is (x, y, w, h)."""
        results = []
        faces = self.detect(image)
        for face in faces[np.argsort(faces[:, 0])]:   # left to right
            box = tuple(int(v) for v in face[:4])
            results.append((box, float(face[-1]), self.embed(image, face)))
        return results


class FaceDatabase:
    """Registered people: several embeddings per name, saved in faces_db/embeddings.npz."""

    def __init__(self, path=DB_FILE):
        self.path = path
        if os.path.exists(path):
            data = np.load(path)
            self.names = list(data['names'])
            self.embeddings = data['embeddings']
        else:
            self.names, self.embeddings = [], np.empty((0, 128), np.float32)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez(self.path, names=np.array(self.names, dtype=str), embeddings=self.embeddings)

    def add(self, name, embeddings):
        self.names += [name] * len(embeddings)
        self.embeddings = np.vstack([self.embeddings, np.asarray(embeddings, np.float32)])

    def remove(self, name):
        keep = [n != name for n in self.names]
        removed = len(keep) - sum(keep)
        self.names = [n for n, k in zip(self.names, keep) if k]
        self.embeddings = self.embeddings[keep]
        return removed

    def people(self):
        """{name: number of registered embeddings}"""
        counts = {}
        for n in self.names:
            counts[n] = counts.get(n, 0) + 1
        return counts

    def identify(self, embedding, threshold):
        """Best match as (name, similarity). name is 'Unknown' when no one is close enough.

        Similarity is cosine similarity (1 = identical); a person's score is their closest
        registered embedding.
        """
        if not self.names:
            return 'Unknown', 0.0
        similarities = self.embeddings @ embedding
        best = int(np.argmax(similarities))
        score = float(similarities[best])
        return (self.names[best] if score >= threshold else 'Unknown'), score
