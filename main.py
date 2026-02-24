"""
Face Detection & Tracking — Clean Kalman SORT
==============================================
Detection: YuNet face detector on downscaled frame (fast)
Tracking:  Kalman SORT with anonymous candidate pool
           Candidates must be seen MIN_HITS times to become a Person_N

Requirements:
    pip install opencv-python numpy
"""

import cv2
import json
import os
import sys
import time
import shutil
import threading
import queue
import urllib.request
import numpy as np
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX = 0
TARGET_FPS = 30
DETECT_WIDTH = 640          # downscale to this for face detection speed
MAX_SKIP = 4
MIN_SKIP = 1

# Tracker
MAX_AGE = 30                # confirmed track: frames without match before death
MIN_HITS = 2                # detections needed for candidate → confirmed
CAND_MAX_AGE = 20           # candidate: frames without match before discard
IOU_THRESH = 0.15           # IoU threshold for matching
CDR_THRESH = 1.5            # center-distance-ratio threshold for fallback matching
SR_THRESH = 4.0             # size-ratio threshold

# Re-ID
REID_SECONDS = 60.0
REID_CDR = 2.0

# Save
SAVE_COOLDOWN = 5.0
DEDUP_IOU = 0.45

# Face detection
FACE_CONF = 0.55
MIN_FACE_PX = 20

# Auditor
AUDIT_INTERVAL = 20
AUDIT_HIST_CORREL = 0.80

# Paths
DATA_FILE = "data.json"
FACES_DIR = "faces"
MODEL_DIR = "models"
DEADZONE = 40

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(p):
    try:
        with open(p) as f: return json.load(f)
    except: return {}

def save_data(p, d):
    with open(p, "w") as f: json.dump(d, f, indent=2)

def dl(url, dest):
    if os.path.exists(dest): return True
    print(f"[DL] {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest); print("[DL] OK"); return True
    except Exception as e:
        print(f"[DL] FAIL: {e}")
        try: os.remove(dest)
        except: pass
        return False

def iou(a, b):
    xa, ya = max(a[0],b[0]), max(a[1],b[1])
    xb, yb = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,xb-xa)*max(0,yb-ya)
    if inter == 0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def clamp_box(x1,y1,x2,y2,w,h):
    return max(0,int(x1)),max(0,int(y1)),min(w,int(x2)),min(h,int(y2))

def scale_boxes(boxes, sx, sy):
    return [(int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)) for x1,y1,x2,y2 in boxes]

def box_center(b): return ((b[0]+b[2])*0.5, (b[1]+b[3])*0.5)
def box_area(b): return max(1.0, float((b[2]-b[0])*(b[3]-b[1])))

def center_dist(a, b):
    """Absolute pixel distance between box centers."""
    ax,ay = box_center(a); bx,by = box_center(b)
    return np.hypot(ax-bx, ay-by)

def cdr(a, b):
    d = center_dist(a, b)
    s = max(a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1], 1.0)
    return d / s

def sr(a, b):
    aa, bb = box_area(a), box_area(b)
    return max(aa,bb)/max(min(aa,bb),1.0)

def hist_of(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv],[0,1],None,[32,32],[0,180,0,256])
    cv2.normalize(h,h); return h


# ═══════════════════════════════════════════════════════════════════════════════
# KALMAN BOX TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

def box_to_z(box):
    w,h = box[2]-box[0], box[3]-box[1]
    return np.array([box[0]+w/2, box[1]+h/2, w*h, w/max(h,1e-6)], dtype=np.float32)

def z_to_box(z):
    w = np.sqrt(max(z[2]*z[3], 1e-6))
    h = max(z[2]/max(w,1e-6), 1e-6)
    return (int(z[0]-w/2), int(z[1]-h/2), int(z[0]+w/2), int(z[1]+h/2))


class KalmanTrack:
    def __init__(self, box):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        self.kf.transitionMatrix[0,4] = 1
        self.kf.transitionMatrix[1,5] = 1
        self.kf.transitionMatrix[2,6] = 1
        self.kf.measurementMatrix = np.zeros((4,7), dtype=np.float32)
        np.fill_diagonal(self.kf.measurementMatrix, 1)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32)
        self.kf.processNoiseCov[4:,4:] *= 0.01
        self.kf.processNoiseCov[:4,:4] *= 10.0
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.errorCovPost[4:,4:] *= 1000.0
        z = box_to_z(box)
        self.kf.statePost[:4,0] = z
        self.kf.statePost[4:,0] = 0
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.last_box = box     # store raw detection box for simple fallback matching

    def predict(self):
        if self.kf.statePost[2,0] + self.kf.statePost[6,0] <= 0:
            self.kf.statePost[6,0] = 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, box):
        self.kf.correct(box_to_z(box).reshape(4,1))
        self.hits += 1
        self.time_since_update = 0
        self.last_box = box

    def get_box(self):
        if self.time_since_update > 0:
            return z_to_box(self.kf.statePre[:4,0])
        return z_to_box(self.kf.statePost[:4,0])


# ═══════════════════════════════════════════════════════════════════════════════
# MATCHING — greedy IoU + center-distance
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_match(detections, track_boxes):
    if not track_boxes: return [], list(range(len(detections))), []
    if not detections: return [], [], list(range(len(track_boxes)))

    nd, nt = len(detections), len(track_boxes)
    scores = np.full((nd, nt), -1.0, dtype=np.float32)
    for d in range(nd):
        for t in range(nt):
            iv = iou(detections[d], track_boxes[t])
            c = cdr(detections[d], track_boxes[t])
            s = sr(detections[d], track_boxes[t])
            if iv >= IOU_THRESH or (c <= CDR_THRESH and s <= SR_THRESH):
                scores[d,t] = iv + 0.3 * max(0, 1.0 - c/CDR_THRESH)

    matches, ud, ut = [], set(), set()
    for _ in range(min(nd, nt)):
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        d, t = int(idx[0]), int(idx[1])
        if scores[d,t] < 0: break
        matches.append((d, t)); ud.add(d); ut.add(t)
        scores[d,:] = -1; scores[:,t] = -1

    return matches, [i for i in range(nd) if i not in ud], [i for i in range(nt) if i not in ut]


def force_match_candidates(det_box, candidates):
    """
    Extra-generous matching for candidates.
    Uses raw last_box (not Kalman prediction) with a generous pixel distance.
    Returns index of best matching candidate, or -1.
    """
    best_idx, best_dist = -1, float("inf")
    for i, c in enumerate(candidates):
        # Use the raw last-seen box for more reliable matching
        d = center_dist(det_box, c.last_box)
        # Allow match if centers are within 2x the face width
        face_size = max(c.last_box[2]-c.last_box[0], c.last_box[3]-c.last_box[1], 50)
        if d < face_size * 2.0 and d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


# ═══════════════════════════════════════════════════════════════════════════════
# TRACKER — with candidate pool
# ═══════════════════════════════════════════════════════════════════════════════

class PersonTrack:
    def __init__(self, name, kalman):
        self.name = name
        self.display_name = name
        self.kf = kalman
        self.last_seen = time.time()
        self.last_save = 0.0
        self.saved_once = False

class Tracker:
    def __init__(self, data):
        self.confirmed = []
        self.candidates = []
        self.data = data
        self.just_graduated = []
        self.recently_lost = {}
        mx = 0
        for k in data:
            if k.startswith("Person_"):
                try: mx = max(mx, int(k.split("_")[1]))
                except: pass
        self.next_id = mx + 1

    def _new_name(self):
        n = f"Person_{self.next_id}"; self.next_id += 1; return n

    def _try_reuse(self, box):
        now = time.time()
        stale = [n for n,r in self.recently_lost.items() if now-r["ts"]>REID_SECONDS]
        for n in stale: self.recently_lost.pop(n,None)
        active = {t.name for t in self.confirmed}
        best, best_sc = None, -1.0
        for name, rec in self.recently_lost.items():
            if name in active: continue
            c = cdr(box, rec["box"])
            if c <= REID_CDR:
                sc = 1.0 - c/REID_CDR
                if sc > best_sc: best_sc, best = sc, name
        if best: self.recently_lost.pop(best, None)
        return best

    def predict(self):
        for t in self.confirmed: t.kf.predict()
        for c in self.candidates: c.predict()

    def update(self, detections):
        self.just_graduated.clear()

        if not detections and not self.confirmed and not self.candidates:
            return

        # ── Step 1: Match detections → confirmed tracks ──
        conf_boxes = [t.kf.get_box() for t in self.confirmed]
        m1, ud1, ut1 = greedy_match(detections, conf_boxes)

        for d, t in m1:
            self.confirmed[t].kf.update(detections[d])
            self.confirmed[t].last_seen = time.time()

        # ── Step 2: Match remaining detections → candidates ──
        remaining = [detections[d] for d in ud1]

        # First try normal matching
        cand_boxes = [c.get_box() for c in self.candidates]
        m2, ud2, ut2 = greedy_match(remaining, cand_boxes)

        matched_cands = set()
        for d, t in m2:
            self.candidates[t].update(remaining[d])
            matched_cands.add(t)

        # Then try generous force-matching for any still-unmatched detections
        still_unmatched = []
        for d in ud2:
            # Only try force-match against candidates that weren't already matched
            avail = [(i, c) for i, c in enumerate(self.candidates)
                     if i not in matched_cands]
            if avail:
                # Find closest candidate by raw pixel distance
                best_i, best_d = -1, float("inf")
                for orig_i, c in avail:
                    dist = center_dist(remaining[d], c.last_box)
                    face_sz = max(c.last_box[2]-c.last_box[0],
                                  c.last_box[3]-c.last_box[1], 50)
                    if dist < face_sz * 2.5 and dist < best_d:
                        best_d, best_i = dist, orig_i
                if best_i >= 0:
                    self.candidates[best_i].update(remaining[d])
                    matched_cands.add(best_i)
                else:
                    still_unmatched.append(d)
            else:
                still_unmatched.append(d)

        # ── Step 3: Create new anonymous candidates for truly unmatched ──
        for d in still_unmatched:
            self.candidates.append(KalmanTrack(remaining[d]))

        # ── Step 4: Graduate candidates that reached MIN_HITS ──
        alive_cands = []
        for c in self.candidates:
            if c.hits >= MIN_HITS:
                box = c.get_box()
                name = self._try_reuse(box) or self._new_name()
                pt = PersonTrack(name, c)
                pt.display_name = self.data.get(name,{}).get("display_name", name)
                if name in self.data and self.data[name].get("image_count",0) > 0:
                    pt.saved_once = True
                self.confirmed.append(pt)
                if name not in self.data:
                    self.data[name] = {
                        "image_count": 0, "first_seen": time.time(),
                        "display_name": name,
                    }
                os.makedirs(os.path.join(FACES_DIR, name), exist_ok=True)
                self.just_graduated.append(pt)
                print(f"[CONFIRMED] {name} (hits={c.hits})")
            elif c.time_since_update < CAND_MAX_AGE:
                alive_cands.append(c)
            # else: candidate expired silently — zero cost
        self.candidates = alive_cands

        # ── Step 5: Remove dead confirmed tracks ──
        alive, dead = [], []
        for t in self.confirmed:
            if t.kf.time_since_update < MAX_AGE:
                alive.append(t)
            else:
                dead.append(t)
        for t in dead:
            self.recently_lost[t.name] = {"box": t.kf.get_box(), "ts": time.time()}
        self.confirmed = alive

        # ── Step 6: Dedup ──
        self._dedup()

    def get_display(self):
        return self.confirmed

    def _dedup(self):
        if len(self.confirmed) < 2: return
        rm = set()
        for i in range(len(self.confirmed)):
            if i in rm: continue
            for j in range(i+1, len(self.confirmed)):
                if j in rm: continue
                if iou(self.confirmed[i].kf.get_box(),
                       self.confirmed[j].kf.get_box()) > DEDUP_IOU:
                    vi = j if self.confirmed[j].kf.hits < self.confirmed[i].kf.hits else i
                    rm.add(vi)
                    v = self.confirmed[vi].name
                    print(f"[DEDUP] {v}")
                    shutil.rmtree(os.path.join(FACES_DIR,v), ignore_errors=True)
                    self.data.pop(v, None)
        if rm:
            self.confirmed = [t for i,t in enumerate(self.confirmed) if i not in rm]


# ═══════════════════════════════════════════════════════════════════════════════
# THREADED CAMERA
# ═══════════════════════════════════════════════════════════════════════════════

class Cam:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened(): raise RuntimeError(f"Cannot open camera {src}")
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.on = True
        threading.Thread(target=self._run, daemon=True).start()
    def _run(self):
        while self.on:
            r, f = self.cap.read()
            with self.lock: self.ret, self.frame = r, f
    def read(self):
        with self.lock:
            return (self.ret, self.frame.copy()) if self.frame is not None else (False, None)
    def release(self):
        self.on = False; time.sleep(0.1); self.cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
# FACE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    """
    Thread-safe face detector. OpenCV DNN is NOT thread-safe, so all calls
    go through a lock. The auditor thread and main thread can both call
    detect() safely.
    """
    def __init__(self):
        self.impl = None
        self._lock = threading.Lock()  # protects all DNN calls
        os.makedirs(MODEL_DIR, exist_ok=True)

        yp = os.path.join(MODEL_DIR, "yunet_2023mar.onnx")
        dl(YUNET_URL, yp)
        if hasattr(cv2, "FaceDetectorYN") and os.path.exists(yp):
            try:
                self.yunet = cv2.FaceDetectorYN.create(yp,"", (320,320), FACE_CONF, 0.3, 5000)
                self.impl = "YuNet"
                print("[INIT] Face detector: YuNet ✓")
                return
            except Exception as e:
                print(f"[WARN] YuNet: {e}")

        pp = os.path.join(MODEL_DIR, "face_deploy.prototxt")
        mp = os.path.join(MODEL_DIR, "res10_face.caffemodel")
        dl(FACE_PROTO_URL, pp); dl(FACE_MODEL_URL, mp)
        if os.path.exists(pp) and os.path.exists(mp):
            try:
                self.net = cv2.dnn.readNetFromCaffe(pp, mp)
                self.impl = "Res10-SSD"
                print("[INIT] Face detector: Res10-SSD ✓")
                return
            except Exception as e:
                print(f"[WARN] Res10: {e}")
        print("[ERROR] No face detector!"); sys.exit(1)

    def detect(self, frame):
        """Thread-safe detection. Returns list of (x1,y1,x2,y2) boxes."""
        if frame is None or frame.size == 0:
            return []
        # Ensure frame is contiguous (required by OpenCV DNN)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        with self._lock:
            try:
                if self.impl == "YuNet":
                    return self._yunet(frame)
                return self._res10(frame)
            except cv2.error as e:
                # Transient DNN errors — log and return empty
                print(f"[WARN] Face detect error: {e}")
                return []

    def _yunet(self, frame):
        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return []
        self.yunet.setInputSize((w, h))
        _, out = self.yunet.detect(frame)
        if out is None: return []
        boxes = []
        for row in out:
            x,y,bw,bh = row[:4]; score = float(row[-1])
            if score < FACE_CONF: continue
            x1,y1,x2,y2 = clamp_box(x, y, x+bw, y+bh, w, h)
            if x2-x1>=MIN_FACE_PX and y2-y1>=MIN_FACE_PX:
                boxes.append((x1,y1,x2,y2))
        return boxes

    def _res10(self, frame):
        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            return []
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0,177.0,123.0))
        self.net.setInput(blob)
        out = self.net.forward()
        boxes = []
        for i in range(out.shape[2]):
            if out[0,0,i,2] < FACE_CONF: continue
            b = (out[0,0,i,3:7]*[w,h,w,h]).astype(int)
            x1,y1,x2,y2 = clamp_box(b[0],b[1],b[2],b[3],w,h)
            if x2-x1>=MIN_FACE_PX and y2-y1>=MIN_FACE_PX:
                boxes.append((x1,y1,x2,y2))
        return boxes


# ═══════════════════════════════════════════════════════════════════════════════
# NMS
# ═══════════════════════════════════════════════════════════════════════════════

def nms(boxes, thresh=0.35):
    if len(boxes) <= 1: return boxes
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    kept = []; used = [False]*len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        bx = list(boxes[i]); cnt = 1
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            if iou(boxes[i], boxes[j]) > thresh:
                used[j] = True
                for k in range(4): bx[k] += boxes[j][k]
                cnt += 1
        used[i] = True
        kept.append(tuple(v//cnt for v in bx))
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# AUDITOR
# ═══════════════════════════════════════════════════════════════════════════════

class Auditor:
    def __init__(self, face_det, data, lock):
        self.det = face_det; self.data = data; self.lock = lock; self.on = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[AUDIT] Background auditor (every {AUDIT_INTERVAL}s)")
    def stop(self): self.on = False
    def _loop(self):
        time.sleep(AUDIT_INTERVAL)
        while self.on:
            try: self._run()
            except Exception as e: print(f"[AUDIT] err: {e}")
            for _ in range(AUDIT_INTERVAL*10):
                if not self.on: return
                time.sleep(0.1)
    def _run(self):
        if not os.path.isdir(FACES_DIR): return
        dirs = [d for d in os.listdir(FACES_DIR)
                if os.path.isdir(os.path.join(FACES_DIR,d)) and d.startswith("Person_")]
        if not dirs: return
        print(f"[AUDIT] Checking {len(dirs)} people...")
        valid = {}; to_del = []
        for pn in dirs:
            pd = os.path.join(FACES_DIR, pn)
            imgs = [f for f in os.listdir(pd) if f.lower().endswith((".jpg",".png",".jpeg"))]
            if not imgs: to_del.append(pn); continue
            real = 0; pdata = []
            for imf in imgs:
                img = cv2.imread(os.path.join(pd, imf))
                if img is None: continue
                if self.det.detect(img): real += 1; pdata.append(hist_of(img))
            if real == 0: to_del.append(pn)
            elif pdata: valid[pn] = pdata
        for pn in to_del:
            print(f"[AUDIT] Delete {pn}")
            shutil.rmtree(os.path.join(FACES_DIR,pn), ignore_errors=True)
            with self.lock: self.data.pop(pn,None); save_data(DATA_FILE,self.data)
        names = sorted(valid.keys(), key=lambda n: int(n.split("_")[1]))
        merged = set()
        for i in range(len(names)):
            if names[i] in merged: continue
            for j in range(i+1, len(names)):
                if names[j] in merged: continue
                bc = max(cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)
                         for h1 in valid[names[i]] for h2 in valid[names[j]])
                if bc >= AUDIT_HIST_CORREL:
                    victim, keeper = names[j], names[i]
                    print(f"[AUDIT] Merge {victim}→{keeper}")
                    merged.add(victim)
                    vd,kd = os.path.join(FACES_DIR,victim), os.path.join(FACES_DIR,keeper)
                    if os.path.isdir(vd):
                        with self.lock:
                            kc = self.data.get(keeper,{}).get("image_count",0)
                            for f in os.listdir(vd):
                                if f.lower().endswith((".jpg",".png",".jpeg")):
                                    kc += 1
                                    try: shutil.move(os.path.join(vd,f),
                                                     os.path.join(kd,f"{keeper}_{kc}.jpg"))
                                    except: pass
                            if keeper in self.data: self.data[keeper]["image_count"]=kc
                            self.data.pop(victim,None); save_data(DATA_FILE,self.data)
                        shutil.rmtree(vd, ignore_errors=True)
        nd, nm = len(to_del), len(merged)
        if nd or nm: print(f"[AUDIT] Cleaned {nd} deleted, {nm} merged")
        else: print("[AUDIT] All clean ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / DRAW / NAME
# ═══════════════════════════════════════════════════════════════════════════════

def save_photo(frame, track, data, lock):
    if track.saved_once: return
    box = track.kf.get_box()
    x1,y1,x2,y2 = box; hf,wf = frame.shape[:2]
    bw,bh = x2-x1, y2-y1
    mx,my = int(bw*0.35), int(bh*0.35)
    c = (max(0,x1-mx),max(0,y1-my),min(wf,x2+mx),min(hf,y2+my))
    crop = frame[c[1]:c[3],c[0]:c[2]]
    if crop.size == 0: return
    with lock:
        data[track.name]["image_count"] = data[track.name].get("image_count",0)+1
        cnt = data[track.name]["image_count"]; save_data(DATA_FILE,data)
    fp = os.path.join(FACES_DIR, track.name, f"{track.name}_{cnt}.jpg")
    cv2.imwrite(fp, crop)
    track.saved_once = True; track.last_save = time.time()
    print(f"[SAVE] {fp}")


class NamePrompter:
    def __init__(self, data, lock):
        self.data=data; self.lock=lock; self.q=queue.Queue()
        self.pending=set(); self.on=True
        threading.Thread(target=self._loop, daemon=True).start()
    def request(self, pk):
        with self.lock: lbl = self.data.get(pk,{}).get("display_name",pk)
        if lbl != pk or pk in self.pending: return
        self.pending.add(pk); self.q.put(pk)
    def stop(self): self.on=False; self.q.put(None)
    def _loop(self):
        while self.on:
            pk = self.q.get()
            if pk is None: return
            try: txt = input(f"[NAME] Enter name for {pk} (blank=keep): ").strip()
            except EOFError: return
            finally: self.pending.discard(pk)
            if not txt: continue
            with self.lock:
                if pk in self.data: self.data[pk]["display_name"]=txt; save_data(DATA_FILE,self.data)
            print(f"[NAME] {pk} → {txt}")


def draw(frame, tracks, face_boxes, fps, skip, n_cand):
    h, w = frame.shape[:2]
    for t in tracks:
        x1,y1,x2,y2 = t.kf.get_box()
        x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(w,x2),min(h,y2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,230,0),2,cv2.LINE_AA)
        lbl = t.display_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw,th),_ = cv2.getTextSize(lbl,font,0.55,1)
        cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+6,y1),(0,200,0),-1,cv2.LINE_AA)
        cv2.putText(frame,lbl,(x1+3,y1-5),font,0.55,(0,0,0),1,cv2.LINE_AA)
    for x1,y1,x2,y2 in face_boxes:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,180,0),1,cv2.LINE_AA)
    hud = (f"Tracked:{len(tracks)}  Cands:{n_cand}  "
           f"Faces:{len(face_boxes)}  FPS:{fps:.0f}  Skip:{skip}")
    cv2.putText(frame,hud,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(frame,hud,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    cx,cy = w//2,h//2
    cv2.rectangle(frame,(cx-DEADZONE,cy-DEADZONE),(cx+DEADZONE,cy+DEADZONE),(0,0,200),1)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    data = load_data(DATA_FILE)
    lock = threading.Lock()
    os.makedirs(FACES_DIR, exist_ok=True)

    face_det = FaceDetector()
    tracker = Tracker(data)
    auditor = Auditor(face_det, data, lock)
    prompter = NamePrompter(data, lock) if sys.stdin.isatty() else None

    try: cam = Cam(CAMERA_INDEX)
    except RuntimeError as e: print(f"ERROR: {e}"); sys.exit(1)
    _, test = cam.read()
    if test is None: print("ERROR: No frame"); sys.exit(1)
    oh, ow = test.shape[:2]

    # Compute downscale for face detection
    if ow > DETECT_WIDTH:
        det_scale = DETECT_WIDTH / ow
        dw, dh = DETECT_WIDTH, int(oh * det_scale)
        sx, sy = ow / dw, oh / dh
        print(f"Camera {CAMERA_INDEX}: {ow}x{oh} → face detect at {dw}x{dh}")
    else:
        dw, dh = ow, oh
        sx, sy = 1.0, 1.0
        print(f"Camera {CAMERA_INDEX}: {ow}x{oh} (no downscale needed)")
    print("Press 'q' to quit.\n")

    skip = 2; fc = 0; fpsd = deque(maxlen=60)
    face_boxes_display = []

    try:
        while True:
            t0 = time.time()
            ret, frame = cam.read()
            if not ret or frame is None: time.sleep(0.001); continue
            fc += 1
            do_det = (fc % skip == 0)

            # Predict every frame — Kalman keeps boxes moving smoothly
            tracker.predict()

            if do_det:
                td0 = time.time()

                # Downscale for detection speed
                if sx != 1.0:
                    small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
                    small = np.ascontiguousarray(small)
                    raw_faces = face_det.detect(small)
                    # Scale boxes back to original resolution
                    face_boxes_orig = scale_boxes(raw_faces, sx, sy)
                else:
                    face_boxes_orig = face_det.detect(frame)

                face_boxes_display = nms(face_boxes_orig)

                # Feed to tracker
                tracker.update(face_boxes_display)

                # Auto-tune skip
                det_ms = (time.time()-td0)*1000
                budget = 1000.0/TARGET_FPS
                if det_ms > budget*0.5: skip = min(skip+1, MAX_SKIP)
                elif det_ms < budget*0.2 and skip > MIN_SKIP: skip -= 1

                # Save photo + name prompt for newly graduated tracks
                for pt in tracker.just_graduated:
                    save_photo(frame, pt, data, lock)
                    if prompter: prompter.request(pt.name)

            # Refresh display
            display = tracker.get_display()
            with lock:
                for t in display:
                    t.display_name = data.get(t.name,{}).get("display_name",t.name)

            el = time.time()-t0
            fpsd.append(el)
            fps = len(fpsd)/max(sum(fpsd),0.001)

            draw(frame, display, face_boxes_display, fps, skip, len(tracker.candidates))
            cv2.imshow("Face Tracker", frame)

            wait = max(1, int(1000.0/TARGET_FPS - el*1000))
            if cv2.waitKey(wait) & 0xFF == ord("q"): break

    finally:
        auditor.stop()
        if prompter: prompter.stop()
        with lock: save_data(DATA_FILE, data)
        cam.release(); cv2.destroyAllWindows()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()