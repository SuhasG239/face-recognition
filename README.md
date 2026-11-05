# Face Recognition

Recognizes people live on a webcam (or in photos and videos), plus a notebook that
trains and evaluates the recognition model.

Register someone with a few webcam shots or photos, and the app draws a box around
each face with that person's name, or **Unknown** for anyone not registered. Adding
people takes seconds and needs no retraining.

Everything runs locally and is free: two pretrained models from the
[OpenCV model zoo](https://github.com/opencv/opencv_zoo), with no API keys or accounts.
Registered faces never leave your computer.

## How it works

1. **Find faces**: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
   detects each face and five landmarks (eyes, nose, mouth corners).
2. **Align**: the face is rotated and cropped to 112x112 using the landmarks.
3. **Embed**: [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
   turns the face into 128 numbers. Photos of the same person give similar numbers.
4. **Match**: the face gets the name of the most similar registered face, if the
   similarity is above the threshold chosen in the notebook (0.46); otherwise it's Unknown.

## Results (face_recognition_model.ipynb)

Tested on [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/): 3,023 photos
of the 62 people with at least 20 photos each, 25% held out for testing.

| Method | Accuracy | Macro F1 |
|---|---|---|
| Eigenfaces (PCA + SVM on pixels) | 61.0% | 0.53 |
| **SFace embeddings + nearest neighbor** (what the app uses) | **99.7%** | **0.995** |
| SFace embeddings + logistic regression / linear SVM | 99.7% | 0.995 |

- **Photos needed**: even 1 registered photo per person gets 99.6% on LFW; 5-10
  helps more in real life, where lighting and angles vary more.
- **Unknown threshold**: with 42 people registered and 20 strangers, a similarity
  cutoff of 0.46 names 99.9% of registered people correctly and wrongly names 0.8% of strangers.
- **Caveat**: these numbers are optimistic. LFW is the benchmark SFace was built
  for, its training data likely includes these public figures, and press photos are
  well lit. Expect more mistakes with dim webcams, side views, or look-alikes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The two model files (39 MB) download into `models/` the first time you run anything.

## Using the app

```bash
python face_app.py register "Suhas" --webcam              # 10 shots; turn your head a little between them
python face_app.py register "Alex" --photos C:\path\to\alex_photos
python face_app.py run                                    # live webcam, press q to quit
python face_app.py run --source group.jpg --save labeled.jpg
python face_app.py run --source clip.mp4 --save labeled.mp4
python face_app.py list
python face_app.py remove "Alex"
```

Each label shows a similarity score. If you show up as Unknown, register more shots
or lower the cutoff (`run --threshold 0.4`). If strangers get your name, raise it.
Registering the same person again adds to their faces.

Registered faces are stored in `faces_db/` (embeddings plus the aligned crops, so you
can see what was captured). That folder is in `.gitignore`. Get people's OK before
registering them.

## Files

- `face_core.py`: face detection, embeddings, and the registered-faces database.
- `face_app.py`: the command-line app (register, run, list, remove).
- `face_recognition_model.ipynb`: training, evaluation, and the threshold choice.
  Kernel: "Python (Face Recognition)". It downloads LFW (about 230 MB) into `data/`.
- `recognition_config.json`: the threshold the notebook chose, read by the app.

## Limitations

- No liveness check: a photo of a registered person held up to the camera will be
  recognized. Fine for a demo, not for security.
- Very dark rooms, faces turned far to the side, masks, or heavy blur reduce accuracy.
- Text labels support plain ASCII letters only.
