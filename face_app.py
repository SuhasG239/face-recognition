"""Register people and recognize them live on the webcam, or in photos and videos.

    python face_app.py register "Suhas" --webcam           # capture 10 shots from the webcam
    python face_app.py register "Alex" --photos alex_pics   # or use a folder of photos
    python face_app.py run                                  # live webcam recognition (press q to quit)
    python face_app.py run --source group.jpg --save labeled.jpg
    python face_app.py list
    python face_app.py remove "Alex"

Registered faces are stored only on this computer, in faces_db/.
"""
import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

from face_core import DB_DIR, FaceDatabase, FaceEngine, load_threshold, shrink

IMAGE_TYPES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
WINDOW = 'Face Recognition (press q to quit)'


def open_camera(index=0):
    """Open a webcam. DirectShow starts much faster on Windows."""
    backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        raise SystemExit(f'Could not open webcam {index}. Is another app using it?')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def folder_name(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_') or 'person'


def save_crops(name, crops):
    """Keep the aligned face crops so you can see what was registered."""
    folder = os.path.join(DB_DIR, folder_name(name))
    os.makedirs(folder, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for k, crop in enumerate(crops):
        cv2.imwrite(os.path.join(folder, f'{stamp}_{k:02d}.jpg'), crop)
    return folder


def largest_face(faces):
    return faces[np.argmax(faces[:, 2] * faces[:, 3])]


def register_from_photos(engine, folder):
    files = sorted(f for f in glob.glob(os.path.join(folder, '*')) if f.lower().endswith(IMAGE_TYPES))
    if not files:
        raise SystemExit(f'No photos found in {folder}')
    embeddings, crops = [], []
    for path in files:
        image = cv2.imread(path)
        if image is None:
            print(f'  skipped {os.path.basename(path)}: not a readable image')
            continue
        image = shrink(image)
        faces = engine.detect(image)
        if len(faces) == 0:
            print(f'  skipped {os.path.basename(path)}: no face found')
            continue
        if len(faces) > 1:
            print(f'  {os.path.basename(path)}: {len(faces)} faces, using the largest')
        face = largest_face(faces)
        embeddings.append(engine.embed(image, face))
        crops.append(engine.align(image, face))
    return embeddings, crops


def register_from_webcam(engine, count, camera_index):
    """Capture a shot every half second while exactly one face is clearly visible."""
    camera = open_camera(camera_index)
    embeddings, crops = [], []
    last_capture = 0.0
    print('Look at the camera and turn your head slightly between shots. Press q to stop early.')
    try:
        while len(embeddings) < count:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit('The webcam stopped sending frames.')
            frame = cv2.flip(frame, 1)   # mirror view feels natural
            faces = engine.detect(frame)
            display = frame.copy()
            if len(faces) == 1 and faces[0][-1] >= 0.9:
                x, y, w, h = faces[0][:4].astype(int)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 0), 2)
                if time.time() - last_capture > 0.5:
                    embeddings.append(engine.embed(frame, faces[0]))
                    crops.append(engine.align(frame, faces[0]))
                    last_capture = time.time()
                    display[:] = 255 - display   # flash
                message = f'Captured {len(embeddings)}/{count}'
            elif len(faces) > 1:
                message = 'Only one person in view, please'
            else:
                message = 'Face the camera'
            cv2.putText(display, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.imshow('Registering (press q to stop)', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return embeddings, crops


def register(args):
    engine = FaceEngine()
    if args.webcam:
        embeddings, crops = register_from_webcam(engine, args.count, args.camera)
    else:
        embeddings, crops = register_from_photos(engine, args.photos)
    if not embeddings:
        raise SystemExit('No faces captured, nothing registered.')
    db = FaceDatabase()
    db.add(args.name, embeddings)
    db.save()
    folder = save_crops(args.name, crops)
    print(f'Registered {len(embeddings)} face(s) for {args.name} '
          f'({db.people()[args.name]} total). Crops saved in {folder}')
    if len(embeddings) < 5:
        print('Tip: 5-10 photos with different angles and lighting make recognition more reliable.')


def label_faces(engine, db, image, threshold):
    """Draw a box and name on every face. Returns the labels found."""
    found = []
    for (x, y, w, h), _, embedding in engine.analyze(image):
        name, score = db.identify(embedding, threshold)
        color = (0, 200, 0) if name != 'Unknown' else (0, 0, 230)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x, max(0, y - th - 12)), (x + tw + 8, y), color, -1)
        cv2.putText(image, text, (x + 4, max(th + 4, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        found.append((name, score))
    return found


def run(args):
    engine, db = FaceEngine(), FaceDatabase()
    threshold = args.threshold if args.threshold is not None else load_threshold()
    if not db.names:
        print('Nobody is registered yet, so every face will show as Unknown. '
              'Register someone first: python face_app.py register "Name" --webcam')
    print(f'Similarity threshold: {threshold} (raise it if strangers get named, lower it if you show as Unknown)')

    source = args.source
    if source is not None and source.lower().endswith(IMAGE_TYPES):
        image = cv2.imread(source)
        if image is None:
            raise SystemExit(f'Could not read {source}')
        image = shrink(image)
        for name, score in label_faces(engine, db, image, threshold):
            print(f'  {name}  (similarity {score:.2f})')
        if args.save:
            cv2.imwrite(args.save, image)
            print(f'Saved {args.save}')
        if not args.no_window:
            cv2.imshow(WINDOW, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    is_webcam = source is None or source.isdigit()
    capture = open_camera(int(source or args.camera)) if is_webcam else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f'Could not open {source}')
    writer = None
    frames, start = 0, time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)
            frame = shrink(frame)
            label_faces(engine, db, frame, threshold)
            frames += 1
            fps = frames / (time.time() - start)
            cv2.putText(frame, f'{fps:.0f} fps', (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
                writer.write(frame)
            if not args.no_window:
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f'Saved {args.save}')
        cv2.destroyAllWindows()


def list_people(args):
    people = FaceDatabase().people()
    if not people:
        print('Nobody registered yet.')
    for name, n in sorted(people.items()):
        print(f'{name}: {n} face(s)')


def remove(args):
    db = FaceDatabase()
    removed = db.remove(args.name)
    if not removed:
        raise SystemExit(f'"{args.name}" is not registered. Registered: {", ".join(db.people()) or "nobody"}')
    db.save()
    print(f'Removed {args.name} ({removed} faces). Their saved crops are in '
          f'{os.path.join(DB_DIR, folder_name(args.name))} if you want to delete those too.')


def main():
    parser = argparse.ArgumentParser(description='Face recognition with a webcam, photos, or videos.')
    commands = parser.add_subparsers(dest='command', required=True)

    p = commands.add_parser('register', help='Add a person from the webcam or a folder of photos')
    p.add_argument('name')
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument('--webcam', action='store_true', help='Capture shots from the webcam')
    how.add_argument('--photos', metavar='FOLDER', help='Folder of photos of this person')
    p.add_argument('--count', type=int, default=10, help='Webcam shots to take (default 10)')
    p.add_argument('--camera', type=int, default=0, help='Webcam number if you have more than one')
    p.set_defaults(func=register)

    p = commands.add_parser('run', help='Recognize faces live, or in a photo or video')
    p.add_argument('--source', help='Photo or video file (default: webcam)')
    p.add_argument('--camera', type=int, default=0, help='Webcam number if you have more than one')
    p.add_argument('--threshold', type=float, help='Similarity needed to name someone (default from the notebook)')
    p.add_argument('--save', metavar='FILE', help='Save the labeled photo or video')
    p.add_argument('--no-window', action='store_true', help='Don\'t open a window (use with --save)')
    p.set_defaults(func=run)

    p = commands.add_parser('list', help='Show registered people')
    p.set_defaults(func=list_people)

    p = commands.add_parser('remove', help='Remove a registered person')
    p.add_argument('name')
    p.set_defaults(func=remove)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
