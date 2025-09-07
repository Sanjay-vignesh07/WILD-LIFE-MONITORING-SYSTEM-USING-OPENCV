# WILD-LIFE-MONITORING-SYSTEM-USING-OPENCV
REAL TIME WILD LIFE TRACKING SYSTEM

An **AI-powered wildlife monitoring prototype** using **YOLOv8 + OpenCV**.  
This project detects and tracks animals from video streams (camera trap footage, CCTV, webcam),  
logs detections to a CSV file, and saves cropped images of each detected animal.

No IoT is required — this runs completely **offline** on your computer.

---

# 🚀 Features
- Real-time animal detection using **YOLOv8 (Ultralytics)**.
- Simple **centroid tracker** to avoid double-counting.
- Logs detections (timestamp, frame number, species, confidence, bounding box).
- Saves **cropped animal images** for later analysis.
- Works with **webcam, video files, or camera trap footage**.

---

# 🛠️ Requirements
Install the following dependencies:

```bash
pip install ultralytics opencv-python pandas

