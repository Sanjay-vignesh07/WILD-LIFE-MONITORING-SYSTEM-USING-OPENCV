"""
wildlife_monitor.py
AI-powered Wildlife Monitoring (prototype, no IoT)

Usage:
    python wildlife_monitor.py --source 0            # webcam
    python wildlife_monitor.py --source video.mp4   # video file

Outputs:
    - detections.csv   (log of detections)
    - crops/           (cropped images of detected animals)
"""

import cv2
import os
import argparse
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# --------- Simple Centroid Tracker (to persist IDs across frames) ----------
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        # next object ID to assign
        self.next_object_id = 0
        # object_id -> centroid
        self.objects = dict()
        # object_id -> number of consecutive frames it has disappeared
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        rects: list of bounding boxes in format (startX, startY, endX, endY)
        returns: dict mapping object_id -> bbox
        """
        if len(rects) == 0:
            # mark all as disappeared
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return {}

        # compute centroids for input rects
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # no existing objects -> register all
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
            # map ids to rects
            return {obj_id: rect for obj_id, rect in zip(range(len(rects)), rects)}

        # build arrays of existing object ids and centroids
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # compute distance matrix between object centroids and input centroids
        D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids[None, :], axis=2)

        # find smallest value in each row then column (Hungarian could be used for optimal)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        assignments = dict()  # object_id -> rect index

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            assignments[object_id] = rects[col]
            used_rows.add(row)
            used_cols.add(col)

        # mark unassigned existing objects as disappeared
        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # register unassigned input centroids as new objects
        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(input_centroids[col])
            new_id = self.next_object_id - 1
            assignments[new_id] = rects[col]

        return assignments

# ----------------- Utility functions -------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def xywh_to_xyxy(x, y, w, h):
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    return x1, y1, x2, y2

# ----------------- Main program ------------------------
def main(args):
    source = args.source
    output_dir = args.output
    ensure_dir(output_dir)
    crops_dir = os.path.join(output_dir, "crops")
    ensure_dir(crops_dir)

    # CSV logging setup
    csv_path = os.path.join(output_dir, "detections.csv")
    if os.path.exists(csv_path):
        df_log = pd.read_csv(csv_path)
    else:
        df_log = pd.DataFrame(columns=["timestamp", "frame_no", "object_id", "class", "confidence", "x1", "y1", "x2", "y2", "crop_path"])

    # load YOLOv8 model (smallest 'n' for speed). Model is downloaded automatically on first run.
    print("[INFO] Loading YOLOv8 model (this may download weights the first time)...")
    model = YOLO("yolov8n.pt")  # change to yolov8m.pt or custom model for better accuracy

    # COCO class names (Ultralytics uses COCO by default)
    coco_names = model.names  # dictionary: id -> name

    # define animal classes of interest (common COCO animals)
    # You can add/remove classes as needed (coco includes: 'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe')
    animal_classes = set(['bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','deer'])

    # NOTE: 'deer' is not in COCO default — if you need species-level detection for specific wildlife species, use custom training datasets.

    # Open video source
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open source:", source)
        return

    tracker = CentroidTracker(max_disappeared=30, max_distance=60)

    frame_no = 0
    print("[INFO] Starting video processing. Press 'q' to quit.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        H, W = frame.shape[:2]

        # optional preprocessing: resize for speed
        in_frame = frame.copy()

        # Run detector (Ultralytics returns results object)
        # For speed: set imgsz and conf in model.predict call
        results = model.predict(source=in_frame, imgsz=640, conf=0.35, iou=0.45, verbose=False)

        # results is list-like; using first (single image)
        detections = []
        if len(results) > 0:
            r = results[0]
            boxes = r.boxes  # Boxes object
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = coco_names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                # get xyxy as ints
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                # filter for animals
                if cls_name in animal_classes:
                    # clamp boxes
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                    detections.append((x1, y1, x2, y2, cls_name, conf))

        # convert to rects for tracker
        rects = [(d[0], d[1], d[2], d[3]) for d in detections]
        assignments = tracker.update(rects)  # mapping object_id -> bbox

        # for reverse lookup: bbox->(class,conf)
        bbox_map = { (d[0],d[1],d[2],d[3]): (d[4], d[5]) for d in detections }

        # draw detections and handle logging
        for obj_id, bbox in assignments.items():
            x1,y1,x2,y2 = bbox
            cls_name, conf = bbox_map.get((x1,y1,x2,y2), ("unknown", 0.0))
            label = f"ID {obj_id} | {cls_name} {conf:.2f}"
            # draw rectangle and label
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Save crop and log once per new object or each frame if you want (here we log every frame detected)
            timestamp = datetime.utcnow().isoformat()
            crop = in_frame[y1:y2, x1:x2]
            crop_filename = f"crop_f{frame_no}_id{obj_id}_{cls_name}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            # avoid empty crops
            if crop.size != 0:
                cv2.imwrite(crop_path, crop)

            # append to dataframe
            new_row = pd.DataFrame([{
                "timestamp": timestamp,
                "frame_no": frame_no,
                "object_id": obj_id,
                "class": cls_name,
                "confidence": float(conf),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "crop_path": crop_path
            }])

            df_log = pd.concat([df_log, new_row], ignore_index=True)


        # show counts on frame
        unique_ids = list(tracker.objects.keys())
        cv2.putText(frame, f"Tracked Objects: {len(unique_ids)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Wildlife Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # periodically flush logs to disk
        if frame_no % 100 == 0:
            df_log.to_csv(csv_path, index=False)
            print(f"[INFO] Flushed logs to {csv_path} at frame {frame_no}")

    # finalization
    df_log.to_csv(csv_path, index=False)
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"[INFO] Done. Processed {frame_no} frames in {elapsed:.2f} seconds. Logs saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="video source (0 for webcam or video file path)")
    parser.add_argument("--output", type=str, default="output", help="output directory for logs and crops")
    args = parser.parse_args()
    main(args)
