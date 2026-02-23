import cv2
import os
import numpy as np
import face_recognition
import mediapipe as mp
import pickle

LIBRARY_PATH = "face_library.pkl"
THRESHOLD = 0.5  # lower = stricter

# Load existing library if it exists
if os.path.exists(LIBRARY_PATH):
    with open(LIBRARY_PATH, "rb") as f:
        face_db = pickle.load(f)
else:
    face_db = {}  # {"Person_1": [embedding1, embedding2], ...}

person_counter = len(face_db) + 1

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

def save_library():
    with open(LIBRARY_PATH, "wb") as f:
        pickle.dump(face_db, f)

def match_face(embedding):
    best_match = None
    best_distance = 1.0

    for name, embeddings in face_db.items():
        for stored in embeddings:
            dist = np.linalg.norm(stored - embedding)
            if dist < best_distance:
                best_distance = dist
                best_match = name

    if best_distance < THRESHOLD:
        return best_match
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)

            face_img = rgb[y1:y2, x1:x2]
            encodings = face_recognition.face_encodings(face_img)

            label = "Detecting..."

            if encodings:
                embedding = encodings[0]
                name = match_face(embedding)

                if name is None:
                    name = f"Person_{person_counter}"
                    face_db[name] = [embedding]
                    person_counter += 1
                    save_library()
                else:
                    face_db[name].append(embedding)

                label = name

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Auto Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
