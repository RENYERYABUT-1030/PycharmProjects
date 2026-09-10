"""
Motion & Face Expression Detector
==================================

Uses your webcam to:
  1. Detect MOTION in the frame (via frame differencing / background subtraction).
  2. Detect FACES (Haar cascade).
  3. Estimate a basic FACIAL EXPRESSION for each detected face:
       - If the FER library (pip install fer) is installed, it is used for
         real deep-learning based emotion recognition (angry, happy, sad,
         surprise, neutral, etc.)
       - Otherwise, a lightweight fallback heuristic uses Haar cascades for
         smile/eye detection to guess between "Happy", "Surprised/Alert",
         and "Neutral".

Controls (while the window is focused):
  q  -> quit
  m  -> toggle motion-detection overlay on/off
  f  -> toggle face/expression detection overlay on/off
  s  -> save a snapshot of the current frame to ./snapshots/

Requirements:
    pip install opencv-python
    # optional, for accurate deep-learning emotion recognition:
    pip install fer tensorflow

Run:
    python motion_face_expression_detector.py
    python motion_face_expression_detector.py --camera 0
"""

import argparse
import os
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Some opencv-python wheels (notably newer 5.x builds) ship without the
# bundled Haar cascade XML files in cv2/data. If that's the case here,
# we transparently download the 3 cascades we need into a local
# "cascades/" folder next to this script and use those instead.
# ----------------------------------------------------------------------
CASCADE_URLS = {
    "haarcascade_frontalface_default.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_smile.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml",
    "haarcascade_eye_tree_eyeglasses.xml":
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
}


def get_cascade_dir():
    """Return a directory that actually contains the Haar cascade XML files,
    downloading them locally if the bundled cv2 data files are missing."""
    bundled_dir = cv2.data.haarcascades
    bundled_ok = all(
        os.path.isfile(os.path.join(bundled_dir, name)) and
        os.path.getsize(os.path.join(bundled_dir, name)) > 0
        for name in CASCADE_URLS
    )
    if bundled_ok:
        return bundled_dir

    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cascades")
    os.makedirs(local_dir, exist_ok=True)

    for name, url in CASCADE_URLS.items():
        path = os.path.join(local_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            print(f"[INFO] Cascade file missing from cv2 install, downloading {name}...")
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"[ERROR] Could not download {name}: {e}")
    return local_dir

# ----------------------------------------------------------------------
# Optional: try to load the FER (Facial Expression Recognition) library
# for real emotion classification. Falls back gracefully if unavailable.
# ----------------------------------------------------------------------
try:
    from fer import FER
    _fer_detector = FER(mtcnn=False)
    HAS_FER = True
except Exception:
    _fer_detector = None
    HAS_FER = False


class MotionDetector:
    """Detects motion between consecutive frames using background subtraction."""

    def __init__(self, history=500, var_threshold=40, min_area=800):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.min_area = min_area

    def detect(self, frame):
        """Returns (motion_detected: bool, boxes: list[(x,y,w,h)], mask)."""
        fg_mask = self.bg_subtractor.apply(frame)
        # Remove shadows (gray value 127) and noise
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8), iterations=2)
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

        return (len(boxes) > 0), boxes, fg_mask


class FaceExpressionDetector:
    """Detects faces and estimates a simple expression label for each."""

    def __init__(self):
        cascade_dir = get_cascade_dir()
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_default.xml"))
        self.smile_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_smile.xml"))
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_eye_tree_eyeglasses.xml"))

        for name, clf in (("face", self.face_cascade),
                           ("smile", self.smile_cascade),
                           ("eye", self.eye_cascade)):
            if clf.empty():
                print(f"[ERROR] Failed to load the '{name}' cascade classifier. "
                      f"Check your internet connection or manually place the "
                      f"XML files in a 'cascades' folder next to this script.")

    def _heuristic_expression(self, roi_gray):
        """Fallback expression guess using smile/eye cascades."""
        smiles = self.smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=8)

        if len(smiles) > 0:
            return "Happy"
        elif len(eyes) >= 2:
            # Rough "eyes wide open, no smile" -> alert/neutral/surprised
            eye_h_avg = np.mean([h for (_, _, _, h) in eyes])
            face_h = roi_gray.shape[0]
            if eye_h_avg / face_h > 0.18:
                return "Surprised"
            return "Neutral"
        else:
            return "Neutral / Eyes closed"

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

        results = []  # list of (x, y, w, h, label, confidence)

        if HAS_FER and len(faces) > 0:
            # Use FER on the full frame (it does its own face detection too,
            # but we align results to our Haar boxes for consistent drawing).
            try:
                fer_results = _fer_detector.detect_emotions(frame)
            except Exception:
                fer_results = []

            for (x, y, w, h) in faces:
                label, conf = "Neutral", 0.0
                # Match the closest FER result to this Haar face box
                best_dist = None
                for fr in fer_results:
                    fx, fy, fw, fh = fr["box"]
                    cx, cy = fx + fw / 2, fy + fh / 2
                    dist = abs(cx - (x + w / 2)) + abs(cy - (y + h / 2))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        emotions = fr["emotions"]
                        top_emotion = max(emotions, key=emotions.get)
                        label = top_emotion.capitalize()
                        conf = emotions[top_emotion]
                results.append((x, y, w, h, label, conf))
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                label = self._heuristic_expression(roi_gray)
                results.append((x, y, w, h, label, None))

        return results


def draw_motion(frame, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
    if boxes:
        cv2.putText(frame, "MOTION DETECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def draw_faces(frame, faces):
    for (x, y, w, h, label, conf) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = label if conf is None else f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Motion & Face Expression Detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--min-motion-area", type=int, default=800,
                         help="Minimum contour area to count as motion")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}. "
              f"Try a different --camera value.")
        return

    motion_detector = MotionDetector(min_area=args.min_motion_area)
    face_detector = FaceExpressionDetector()

    show_motion = True
    show_faces = True

    os.makedirs("snapshots", exist_ok=True)

    print("=" * 60)
    print("Motion & Face Expression Detector running.")
    print(f"FER deep-learning emotion model: {'ENABLED' if HAS_FER else 'not installed (using fallback heuristic)'}")
    print("Press 'q' to quit | 'm' toggle motion | 'f' toggle faces | 's' snapshot")
    print("=" * 60)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame, exiting.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural webcam view
        display = frame.copy()

        if show_motion:
            motion_detected, boxes, _ = motion_detector.detect(frame)
            draw_motion(display, boxes)

        if show_faces:
            faces = face_detector.detect(frame)
            draw_faces(display, faces)

        # FPS counter
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(display, f"FPS: {fps:.1f}", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Motion & Face Expression Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_motion = not show_motion
        elif key == ord('f'):
            show_faces = not show_faces
        elif key == ord('s'):
            fname = f"snapshots/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, display)
            print(f"[INFO] Snapshot saved to {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()