"""
Face Detection & Tracking — SORT-Style Kalman Tracker
======================================================
WHY THIS IS DIFFERENT:

Every previous version used simple centroid distance + EMA smoothing.
That breaks when faces move fast, rotate, or skip between detection frames.

This version uses a Kalman Filter tracker (the same algorithm used in
SORT — Simple Online and Realtime Tracking, the industry standard).

Each tracked face has a Kalman state: [cx, cy, w, h, vx, vy, vw, vh]
  • Predicts WHERE the face will be next frame (velocity-aware)
  • Boxes stay locked on during fast head turns and movement
  • Between detection frames, Kalman prediction keeps boxes smooth
  • Matching uses predicted positions, not stale last-seen positions
  • Handles faces teleporting across screen between detections

Detection: SSD + YuNet (fast, on downscaled frames)
Optional:  RetinaFace in background thread for bonus accuracy
Auditor:   Background thread cleans false positives & merges duplicates

Requirements:
    pip install opencv-python numpy           # minimum (Pi 5 + Mac)
    pip install retina-face                    # optional (best accuracy)
"""

import cv2
import json
import os
import sys
import time
import shutil
import threading
import urllib.request
import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX = 0
TARGET_FPS = 30
DETECT_WIDTH = 420          # downscale to this width for detection speed
MAX_SKIP = 6                # max frames between detections
MIN_SKIP = 1

# Kalman tracker
MAX_AGE = 25                # frames a track survives without a match
MIN_HITS = 3                # detections before a track is shown (confirmation)
IOU_THRESHOLD = 0.25        # minimum IoU for matching detection → track

# Persistence
SAVE_COOLDOWN = 3.0
INACTIVE_TIMEOUT = 1.5      # seconds before marking inactive for photo logic
DEDUP_IOU = 0.5

# Detector thresholds
DNN_CONF = 0.52
YUNET_CONF = 0.65
RETINA_CONF = 0.80
MIN_FACE_PX = 30            # on detect-resolution frame

# Auditor
AUDIT_INTERVAL = 30
AUDIT_HIST_CORREL = 0.80

# Paths
DATA_FILE = "data.json"
FACES_DIR = "faces"
MODEL_DIR = "models"
DEADZONE = 40

PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
CAFFE_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(p):
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except:
            pass
    return {}


def save_data(p, d):
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def dl(url, dest):
    if os.path.exists(dest):
        return
    print(f"[DL] {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("[DL] OK")
    except Exception as e:
        print(f"[DL] FAIL: {e}")
        sys.exit(1)


def iou_single(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


def iou_matrix(dets, trks):
    """Compute IoU matrix between detections and tracks. Shape: (D, T)."""
    m = np.zeros((len(dets), len(trks)), dtype=np.float32)
    for d in range(len(dets)):
        for t in range(len(trks)):
            m[d, t] = iou_single(dets[d], trks[t])
    return m


def clamp(x1, y1, x2, y2, w, h):
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def scale_boxes(boxes, sx, sy):
    return [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
            for x1, y1, x2, y2 in boxes]


def hist_of(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN BOX TRACKER — the core of smooth tracking
# ═══════════════════════════════════════════════════════════════════════════════

def box_to_z(box):
    """[x1,y1,x2,y2] → [cx, cy, area, aspect_ratio]"""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w / 2.0
    cy = box[1] + h / 2.0
    return np.array([cx, cy, w * h, w / max(h, 1e-6)], dtype=np.float32)


def z_to_box(z):
    """[cx, cy, area, aspect_ratio] → [x1, y1, x2, y2]"""
    w = np.sqrt(max(z[2] * z[3], 1e-6))
    h = max(z[2] / max(w, 1e-6), 1e-6)
    return np.array([
        z[0] - w / 2.0,
        z[1] - h / 2.0,
        z[0] + w / 2.0,
        z[1] + h / 2.0,
    ], dtype=np.float32)


class KalmanBoxTracker:
    """
    Kalman filter for a single tracked face.

    State: [cx, cy, area, ratio, v_cx, v_cy, v_area]
      - Tracks position, size, AND velocity
      - Predicts next position even without a detection
      - Handles fast movement naturally through velocity terms

    This is the same approach used in SORT (Bewley et al., 2016).
    """
    _count = 0

    def __init__(self, box, name):
        self.kf = cv2.KalmanFilter(7, 4)  # 7 state dims, 4 measurement dims

        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # Noise covariances (tuned for face tracking)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32)
        self.kf.processNoiseCov[4:, 4:] *= 0.01  # low velocity noise
        self.kf.processNoiseCov[:4, :4] *= 10.0   # higher position noise
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.errorCovPost[4:, 4:] *= 1000.0    # high initial velocity uncertainty

        # Init state from first measurement
        z = box_to_z(box)
        self.kf.statePost[:4, 0] = z
        self.kf.statePost[4:, 0] = 0  # zero initial velocity

        self.name = name
        self.hits = 1           # total successful matches
        self.age = 0            # frames since creation
        self.time_since_update = 0  # frames since last match
        self.last_seen_time = time.time()
        self.last_save_time = 0.0
        self.active = True

    def predict(self):
        """Advance state by one frame. Returns predicted box."""
        # Prevent area from going negative
        if self.kf.statePost[2, 0] + self.kf.statePost[6, 0] <= 0:
            self.kf.statePost[6, 0] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_box()

    def update(self, box):
        """Correct state with a matched detection."""
        z = box_to_z(box)
        self.kf.correct(z.reshape(4, 1))
        self.hits += 1
        self.time_since_update = 0
        self.last_seen_time = time.time()
        was_inactive = not self.active
        self.active = True
        return was_inactive

    def get_box(self):
        """Current estimated box [x1, y1, x2, y2] as ints."""
        state = self.kf.statePost[:4, 0]
        box = z_to_box(state)
        return tuple(int(v) for v in box)


# ═══════════════════════════════════════════════════════════════════════════════
# HUNGARIAN-STYLE MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def match_detections_to_tracks(detections, trackers, iou_threshold=IOU_THRESHOLD):
    """
    Match detections to tracked objects using IoU.
    Returns: (matches, unmatched_dets, unmatched_trks)

    Uses greedy matching on IoU matrix (fast, and optimal enough for <20 faces).
    For strict optimality you'd use scipy.optimize.linear_sum_assignment,
    but greedy works perfectly for typical face counts.
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    iou_mat = iou_matrix(detections, trackers)

    matches = []
    used_d = set()
    used_t = set()

    # Iterate in order of highest IoU
    while True:
        if iou_mat.size == 0:
            break
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        d, t = int(idx[0]), int(idx[1])
        if iou_mat[d, t] < iou_threshold:
            break
        if d not in used_d and t not in used_t:
            matches.append((d, t))
            used_d.add(d)
            used_t.add(t)
        iou_mat[d, t] = 0  # zero out so we pick next best

    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    unmatched_t = [i for i in range(len(trackers)) if i not in used_t]
    return matches, unmatched_d, unmatched_t


# ═══════════════════════════════════════════════════════════════════════════════
# SORT TRACKER (manages all KalmanBoxTrackers)
# ═══════════════════════════════════════════════════════════════════════════════

class SORTTracker:
    def __init__(self, data):
        self.trackers = []       # list of KalmanBoxTracker
        self.data = data
        self.frame_count = 0
        mx = 0
        for k in data:
            if k.startswith("Person_"):
                try:
                    mx = max(mx, int(k.split("_")[1]))
                except:
                    pass
        self.next_id = mx + 1

    def _new_name(self):
        n = f"Person_{self.next_id}"
        self.next_id += 1
        return n

    def predict(self):
        """Predict all trackers forward one step. Call every frame."""
        for trk in self.trackers:
            trk.predict()

    def update(self, detections):
        """
        Update tracker with new detections.
        Call on frames where detection ran.
        Returns list of (KalmanBoxTracker, newly_appeared: bool) for display.
        """
        self.frame_count += 1
        now = time.time()

        # Get predicted boxes for matching
        pred_boxes = [trk.get_box() for trk in self.trackers]

        # Match
        matches, unmatched_d, unmatched_t = match_detections_to_tracks(
            detections, pred_boxes, IOU_THRESHOLD
        )

        results = []

        # Update matched trackers
        for d, t in matches:
            reappeared = self.trackers[t].update(detections[d])
            if self.trackers[t].hits >= MIN_HITS:
                results.append((self.trackers[t], reappeared and self.trackers[t].hits == MIN_HITS))

        # Create new trackers for unmatched detections
        for d in unmatched_d:
            name = self._new_name()
            trk = KalmanBoxTracker(detections[d], name)
            self.trackers.append(trk)
            self.data[name] = {"image_count": 0, "first_seen": time.time()}
            os.makedirs(os.path.join(FACES_DIR, name), exist_ok=True)
            print(f"[NEW] {name}")
            # Don't show until MIN_HITS reached

        # Mark inactive
        for t in unmatched_t:
            if now - self.trackers[t].last_seen_time > INACTIVE_TIMEOUT:
                self.trackers[t].active = False

        # Return confirmed tracks that are alive
        for trk in self.trackers:
            if trk.time_since_update == 0 and trk.hits >= MIN_HITS:
                # Already added in matches loop
                pass
            elif trk.time_since_update < MAX_AGE and trk.hits >= MIN_HITS:
                # Still coasting — show predicted box
                already = any(r[0].name == trk.name for r in results)
                if not already:
                    results.append((trk, False))

        # Remove dead tracks
        dead = [trk for trk in self.trackers if trk.time_since_update >= MAX_AGE]
        for trk in dead:
            if trk.hits < MIN_HITS:
                # Never confirmed — clean up
                d = os.path.join(FACES_DIR, trk.name)
                if os.path.isdir(d):
                    shutil.rmtree(d, ignore_errors=True)
                self.data.pop(trk.name, None)
        self.trackers = [t for t in self.trackers if t.time_since_update < MAX_AGE]

        # De-duplicate overlapping active tracks
        self._dedup()

        return results

    def get_display_tracks(self):
        """Get all confirmed, recently-seen tracks for drawing between detections."""
        return [(trk, False) for trk in self.trackers
                if trk.hits >= MIN_HITS and trk.time_since_update < MAX_AGE]

    def _dedup(self):
        if len(self.trackers) < 2:
            return
        rm = set()
        for i in range(len(self.trackers)):
            if i in rm:
                continue
            for j in range(i + 1, len(self.trackers)):
                if j in rm:
                    continue
                if iou_single(self.trackers[i].get_box(),
                              self.trackers[j].get_box()) > DEDUP_IOU:
                    # Remove the one with fewer hits
                    victim_idx = j if self.trackers[j].hits < self.trackers[i].hits else i
                    rm.add(victim_idx)
                    v = self.trackers[victim_idx].name
                    print(f"[DEDUP] {v}")
                    d = os.path.join(FACES_DIR, v)
                    if os.path.isdir(d):
                        shutil.rmtree(d, ignore_errors=True)
                    self.data.pop(v, None)
        if rm:
            self.trackers = [t for idx, t in enumerate(self.trackers) if idx not in rm]


# ═══════════════════════════════════════════════════════════════════════════════
# THREADED CAMERA
# ═══════════════════════════════════════════════════════════════════════════════

class Cam:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.on = True
        threading.Thread(target=self._go, daemon=True).start()

    def _go(self):
        while self.on:
            r, f = self.cap.read()
            with self.lock:
                self.ret, self.frame = r, f

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def release(self):
        self.on = False
        time.sleep(0.1)
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTORS
# ═══════════════════════════════════════════════════════════════════════════════

class SSD:
    def __init__(self, proto, model):
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        print("[INIT] SSD ResNet-10 ✓")

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        out = self.net.forward()
        boxes = []
        for i in range(out.shape[2]):
            c = out[0, 0, i, 2]
            if c < DNN_CONF:
                continue
            b = (out[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
            x1, y1, x2, y2 = clamp(b[0], b[1], b[2], b[3], w, h)
            if x2 - x1 >= MIN_FACE_PX and y2 - y1 >= MIN_FACE_PX:
                boxes.append((x1, y1, x2, y2))
        return boxes


class YuNetDet:
    def __init__(self, path):
        self.d = cv2.FaceDetectorYN.create(
            path, "", (320, 320), YUNET_CONF, 0.3, 5000)
        print("[INIT] YuNet ✓")

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.d.setInputSize((w, h))
        _, faces = self.d.detect(frame)
        boxes = []
        if faces is not None:
            for f in faces:
                x1, y1 = int(f[0]), int(f[1])
                x2, y2 = x1 + int(f[2]), y1 + int(f[3])
                x1, y1, x2, y2 = clamp(x1, y1, x2, y2, w, h)
                if x2 - x1 >= MIN_FACE_PX and y2 - y1 >= MIN_FACE_PX:
                    boxes.append((x1, y1, x2, y2))
        return boxes


# ═══════════════════════════════════════════════════════════════════════════════
# RETINAFACE BACKGROUND THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class RetinaThread:
    def __init__(self):
        from retinaface import RetinaFace as RF
        self.RF = RF
        self.lock = threading.Lock()
        self.input_frame = None
        self.result_boxes = []
        self.new_input = False
        self.on = True
        self.ready = False
        print("[INIT] RetinaFace loading...")
        tmp = "/tmp/_rf_warm.jpg"
        cv2.imwrite(tmp, np.zeros((100, 100, 3), dtype=np.uint8))
        try:
            self.RF.detect_faces(tmp)
        except:
            pass
        if os.path.exists(tmp):
            os.remove(tmp)
        print("[INIT] RetinaFace ✓ (background)")
        self.ready = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        with self.lock:
            self.input_frame = frame.copy()
            self.new_input = True

    def get_results(self):
        with self.lock:
            return list(self.result_boxes)

    def stop(self):
        self.on = False

    def _loop(self):
        tmp = "/tmp/_rf_det.jpg"
        while self.on:
            frame = None
            with self.lock:
                if self.new_input:
                    frame = self.input_frame
                    self.new_input = False
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                cv2.imwrite(tmp, frame)
                resp = self.RF.detect_faces(tmp)
            except:
                resp = {}
            h, w = frame.shape[:2]
            boxes = []
            if isinstance(resp, dict):
                for key in resp:
                    face = resp[key]
                    if face.get("score", 0) < RETINA_CONF:
                        continue
                    fa = face["facial_area"]
                    x1, y1, x2, y2 = clamp(fa[0], fa[1], fa[2], fa[3], w, h)
                    if x2 - x1 > 10 and y2 - y1 > 10:
                        boxes.append((x1, y1, x2, y2))
            with self.lock:
                self.result_boxes = boxes

    def detect_single(self, img):
        tmp = "/tmp/_rf_aud.jpg"
        cv2.imwrite(tmp, img)
        try:
            resp = self.RF.detect_faces(tmp)
        except:
            return []
        h, w = img.shape[:2]
        boxes = []
        if isinstance(resp, dict):
            for key in resp:
                if resp[key].get("score", 0) >= 0.50:
                    fa = resp[key]["facial_area"]
                    x1, y1, x2, y2 = clamp(fa[0], fa[1], fa[2], fa[3], w, h)
                    if x2 - x1 > 10 and y2 - y1 > 10:
                        boxes.append((x1, y1, x2, y2))
        return boxes


# ═══════════════════════════════════════════════════════════════════════════════
# NMS FUSION
# ═══════════════════════════════════════════════════════════════════════════════

def fuse_boxes(box_lists):
    pool = []
    for bl in box_lists:
        pool.extend(bl)
    if not pool:
        return []
    # Sort by area descending
    pool.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    used = [False] * len(pool)
    for i in range(len(pool)):
        if used[i]:
            continue
        bx = list(pool[i])
        cnt = 1
        for j in range(i + 1, len(pool)):
            if used[j]:
                continue
            if iou_single(pool[i], pool[j]) > 0.35:
                used[j] = True
                for k in range(4):
                    bx[k] += pool[j][k]
                cnt += 1
        used[i] = True
        kept.append(tuple(v // cnt for v in bx))
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# AUDITOR
# ═══════════════════════════════════════════════════════════════════════════════

class Auditor:
    def __init__(self, detect_fn, data, lock):
        self.detect = detect_fn
        self.data = data
        self.lock = lock
        self.on = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[AUDIT] Started (every 30s)")

    def stop(self):
        self.on = False

    def _loop(self):
        time.sleep(15)
        while self.on:
            try:
                self._run()
            except Exception as e:
                print(f"[AUDIT] err: {e}")
            for _ in range(AUDIT_INTERVAL * 10):
                if not self.on:
                    return
                time.sleep(0.1)

    def _run(self):
        if not os.path.isdir(FACES_DIR):
            return
        dirs = [d for d in os.listdir(FACES_DIR)
                if os.path.isdir(os.path.join(FACES_DIR, d))
                and d.startswith("Person_")]
        if not dirs:
            return
        print(f"[AUDIT] Checking {len(dirs)} people...")

        valid = {}
        to_del = []

        for pn in dirs:
            pd = os.path.join(FACES_DIR, pn)
            imgs = [f for f in os.listdir(pd)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            if not imgs:
                to_del.append(pn)
                continue
            real = 0
            pdata = []
            for imf in imgs:
                img = cv2.imread(os.path.join(pd, imf))
                if img is None:
                    continue
                if self.detect(img):
                    real += 1
                    pdata.append((img, hist_of(img)))
            if real == 0:
                to_del.append(pn)
            elif pdata:
                valid[pn] = pdata

        for pn in to_del:
            print(f"[AUDIT] Delete {pn} (no faces)")
            shutil.rmtree(os.path.join(FACES_DIR, pn), ignore_errors=True)
            with self.lock:
                self.data.pop(pn, None)
                save_data(DATA_FILE, self.data)

        names = sorted(valid.keys(), key=lambda n: int(n.split("_")[1]))
        merged = set()
        for i in range(len(names)):
            if names[i] in merged:
                continue
            for j in range(i + 1, len(names)):
                if names[j] in merged:
                    continue
                bc = max(
                    cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                    for _, h1 in valid[names[i]]
                    for _, h2 in valid[names[j]]
                )
                if bc >= AUDIT_HIST_CORREL:
                    victim, keeper = names[j], names[i]
                    print(f"[AUDIT] Merge {victim}→{keeper} (sim={bc:.2f})")
                    merged.add(victim)
                    vd = os.path.join(FACES_DIR, victim)
                    kd = os.path.join(FACES_DIR, keeper)
                    if os.path.isdir(vd):
                        with self.lock:
                            kc = self.data.get(keeper, {}).get("image_count", 0)
                            for f in os.listdir(vd):
                                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                                    kc += 1
                                    try:
                                        shutil.move(
                                            os.path.join(vd, f),
                                            os.path.join(kd, f"{keeper}_{kc}.jpg"))
                                    except:
                                        pass
                            if keeper in self.data:
                                self.data[keeper]["image_count"] = kc
                            self.data.pop(victim, None)
                            save_data(DATA_FILE, self.data)
                        shutil.rmtree(vd, ignore_errors=True)

        nd, nm = len(to_del), len(merged)
        if nd or nm:
            print(f"[AUDIT] Done: deleted {nd}, merged {nm}")
        else:
            print("[AUDIT] All clean ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / DRAW
# ═══════════════════════════════════════════════════════════════════════════════

def save_photo(frame, trk, data, lock):
    now = time.time()
    if now - trk.last_save_time < SAVE_COOLDOWN:
        return
    box = trk.get_box()
    x1, y1, x2, y2 = box
    hf, wf = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * 0.3), int(bh * 0.3)
    c = (max(0, x1 - mx), max(0, y1 - my), min(wf, x2 + mx), min(hf, y2 + my))
    crop = frame[c[1]:c[3], c[0]:c[2]]
    if crop.size == 0:
        return
    with lock:
        data[trk.name]["image_count"] = data[trk.name].get("image_count", 0) + 1
        cnt = data[trk.name]["image_count"]
        save_data(DATA_FILE, data)
    fp = os.path.join(FACES_DIR, trk.name, f"{trk.name}_{cnt}.jpg")
    cv2.imwrite(fp, crop)
    trk.last_save_time = now
    print(f"[SAVE] {fp}")


def draw_frame(frame, tracks, n_det, fps, skip, backend):
    h, w = frame.shape[:2]

    for trk, _ in tracks:
        x1, y1, x2, y2 = trk.get_box()
        # Clamp to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Green box with slight transparency effect via thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2, cv2.LINE_AA)

        # Label background
        lbl = trk.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(lbl, font, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1),
                      (0, 200, 0), -1, cv2.LINE_AA)
        cv2.putText(frame, lbl, (x1 + 3, y1 - 5),
                    font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # HUD
    hud = f"Faces:{n_det}  FPS:{fps:.0f}  Skip:{skip}  [{backend}]"
    cv2.putText(frame, hud, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, hud, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Center deadzone
    cx, cy = w // 2, h // 2
    cv2.rectangle(frame, (cx - DEADZONE, cy - DEADZONE),
                  (cx + DEADZONE, cy + DEADZONE), (0, 0, 200), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    data = load_data(DATA_FILE)
    lock = threading.Lock()
    os.makedirs(FACES_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download models
    pp = os.path.join(MODEL_DIR, "deploy.prototxt")
    cm = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
    yp = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
    dl(PROTOTXT_URL, pp)
    dl(CAFFE_URL, cm)
    dl(YUNET_URL, yp)

    # Fast detectors
    fast = []
    names = []
    try:
        fast.append(SSD(pp, cm))
        names.append("SSD")
    except Exception as e:
        print(f"[WARN] SSD: {e}")
    if hasattr(cv2, "FaceDetectorYN"):
        try:
            fast.append(YuNetDet(yp))
            names.append("YuNet")
        except Exception as e:
            print(f"[WARN] YuNet: {e}")
    if not fast:
        cp = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cp):
            cas = cv2.CascadeClassifier(cp)

            class HW:
                def detect(self, f):
                    g = cv2.equalizeHist(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                    h_, w_ = f.shape[:2]
                    return [(x, y, x + bw, y + bh)
                            for x, y, bw, bh in
                            cas.detectMultiScale(g, 1.1, 5, 0, (60, 60))]

            fast.append(HW())
            names.append("Haar")
        else:
            print("ERROR: No detector")
            sys.exit(1)

    # Optional RetinaFace
    retina = None
    try:
        retina = RetinaThread()
        names.append("RetinaFace(bg)")
    except ImportError:
        print("[INFO] retina-face not installed — SSD+YuNet only")
        print("       For best accuracy: pip install retina-face")
    except Exception as e:
        print(f"[INFO] RetinaFace unavailable: {e}")

    backend = " + ".join(names)
    print(f"\n[INIT] Detectors: {backend}")

    def audit_detect(img):
        if retina and retina.ready:
            return retina.detect_single(img)
        return fast[0].detect(img) if fast else []

    # Camera
    try:
        cam = Cam(CAMERA_INDEX)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    _, test = cam.read()
    if test is None:
        print("ERROR: No frame")
        sys.exit(1)
    oh, ow = test.shape[:2]
    sc = DETECT_WIDTH / ow
    dw, dh = DETECT_WIDTH, int(oh * sc)
    sx, sy = ow / dw, oh / dh
    print(f"Camera {CAMERA_INDEX}: {ow}x{oh} → detect at {dw}x{dh}")
    print(f"Press 'q' to quit.\n")

    # Tracker + auditor
    tracker = SORTTracker(data)
    auditor = Auditor(audit_detect, data, lock)

    skip = 2
    fc = 0
    fpsd = deque(maxlen=60)
    n_det = 0
    display_tracks = []
    retina_counter = 0
    retina_interval = 6

    try:
        while True:
            t0 = time.time()
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue
            fc += 1
            do_det = (fc % skip == 0)

            # EVERY frame: Kalman predict (keeps boxes moving smoothly)
            tracker.predict()

            if do_det:
                td0 = time.time()

                small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)

                all_boxes = []
                for det in fast:
                    try:
                        all_boxes.append(scale_boxes(det.detect(small), sx, sy))
                    except:
                        all_boxes.append([])

                # Merge RetinaFace results
                if retina:
                    rb = retina.get_results()
                    if rb:
                        all_boxes.append(rb)
                    retina_counter += 1
                    if retina_counter >= retina_interval:
                        retina_counter = 0
                        retina.submit(frame)

                fused = fuse_boxes(all_boxes)
                n_det = len(fused)
                display_tracks = tracker.update(fused)

                # Adapt skip
                det_ms = (time.time() - td0) * 1000
                budget = 1000.0 / TARGET_FPS
                if det_ms > budget * 0.5:
                    skip = min(skip + 1, MAX_SKIP)
                elif det_ms < budget * 0.2 and skip > MIN_SKIP:
                    skip -= 1

                # Save photos
                for trk, new in display_tracks:
                    if new:
                        save_photo(frame, trk, data, lock)
            else:
                # Between detections: show Kalman-predicted positions
                display_tracks = tracker.get_display_tracks()

            # FPS
            el = time.time() - t0
            fpsd.append(el)
            fps = len(fpsd) / max(sum(fpsd), 0.001)

            draw_frame(frame, display_tracks, n_det, fps, skip, backend)
            cv2.imshow("Face Tracker", frame)

            wait = max(1, int(1000.0 / TARGET_FPS - el * 1000))
            if cv2.waitKey(wait) & 0xFF == ord("q"):
                break

    finally:
        if retina:
            retina.stop()
        auditor.stop()
        with lock:
            save_data(DATA_FILE, data)
        cam.release()
        cv2.destroyAllWindows()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()