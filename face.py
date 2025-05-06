# dlib is optional; we mostly use Haar cascades. Import lazily.
import cv2
try:
    import dlib
except ImportError:
    dlib = None
import numpy as np
import os
import json
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm  # progress bar


def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return [], 0, 0  # No boxes, no removals

    original_count = len(boxes)

    boxes = np.array(boxes).astype("float")
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1],
                                               np.where(overlap > overlapThresh)[0])))

    final_boxes = boxes[pick].astype("int")
    kept_count = len(final_boxes)
    removed_count = original_count - kept_count

    return final_boxes, kept_count, removed_count


def detect_faces_haar(img_gray, face_cascade):
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.02,
        minNeighbors=4,
        minSize=(40, 40)
    )
    return [(x, y, w, h) for (x, y, w, h) in faces]


def detect_faces_dlib(img_gray, hog_face_detector):
    if hog_face_detector is None:
        return []
    dlib_faces = hog_face_detector(img_gray, 1)
    faces = []
    for rect in dlib_faces:
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        faces.append((x, y, w, h))
    return faces

if __name__ == "__main__":
    # --- SETTINGS ---
    OUTPUT_CROPPED_FACES = True   # Set True to save cropped faces
    OUTPUT_DIR = "detected_faces"
    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    DETECTOR = "haar"  # or "dlib"

    # --- PREPARE ---
    if OUTPUT_CROPPED_FACES and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load detectors
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    hog_face_detector = dlib.get_frontal_face_detector()
    if dlib is None:
        hog_face_detector = None

    # Load LFW dataset
    lfw_dataset = fetch_lfw_people(min_faces_per_person=5, resize=1.0)
    images = lfw_dataset.images
    image_shape = images[0].shape

    # Placeholder for bounding boxes
    detection_results = []

    # --- PROCESS IMAGES ---
    print("Starting face detection...")
    total_boxes_before = 0
    total_boxes_after = 0
    total_boxes_removed = 0
    for idx, img in enumerate(tqdm(images)):
        img_uint8 = (img * 255).astype(np.uint8)
        img_colored = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

        # Choose detection method
        if DETECTOR == "haar":
            faces = detect_faces_haar(img_uint8, face_cascade)
        elif DETECTOR == "dlib":
            faces = detect_faces_dlib(img_uint8, hog_face_detector)
        else:
            raise ValueError("Invalid DETECTOR setting!")

        faces, kept, removed = non_max_suppression(faces, overlapThresh=0.3)

        total_boxes_before += kept + removed
        total_boxes_after += kept
        total_boxes_removed += removed

        for i, (x, y, w, h) in enumerate(faces):
            padding = 0.1  # 10% padding
            # Expand the bounding box
            x = max(0, int(x - w * padding))
            y = max(0, int(y - h * padding))
            w = int(w * (1 + 2 * padding))
            h = int(h * (1 + 2 * padding))

            # Make sure we don't go outside image boundaries
            x2 = min(x + w, img_colored.shape[1])
            y2 = min(y + h, img_colored.shape[0])

            # Save bounding box info
            detection_results.append({
                "image_id": idx,
                "bbox": {"x": int(x), "y": int(y), "w": int(x2 - x), "h": int(y2 - y)}
            })

            if OUTPUT_CROPPED_FACES:
                face_crop = img_colored[y:y+h, x:x+w]
                filename = os.path.join(OUTPUT_DIR, f"face_{idx}_{i}.jpg")
                cv2.imwrite(filename, face_crop)

    print(f"\nProcessed {len(images)} images.")
    print(f"Detected {len(detection_results)} faces total.")
    print(f"[NMS Summary] Total boxes removed by NMS: {total_boxes_removed}")

    # OPTIONAL: Save bounding boxes to a file
    with open("bounding_boxes.json", "w") as f:
        json.dump(detection_results, f, indent=4)

    print("\nBounding boxes saved to bounding_boxes.json")

    # --- DONE ---
