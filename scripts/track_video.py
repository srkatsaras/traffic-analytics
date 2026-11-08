import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Define ROOT const and input/output directories
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/vehicles.mp4"
OUTPUT_DIR = ROOT / "outputs/vehicle_tracking"

MAX_FRAMES = 300

detections_logged = 0
seen_track_ids = set()

def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Video not found: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(ROOT / "yolov8n.pt"))

    vehicle_names = {'car', 'motorcycle', 'bus', 'truck'}
    vehicle_ids = [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in vehicle_names
    ]

    cap = cv2.VideoCapture(INPUT)
    writer = None
    frames_processed = 0
    detections_logged = 0

    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise RuntimeError("The video has no usable FPS metadata.")

        video_path = OUTPUT_DIR / "annotated.mp4"
        csv_path = OUTPUT_DIR / "detections.csv"

        print(f"Device: {device}")
        print(f"Input FPS: {fps:.2f}")

        with csv_path.open('w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame_index", "time_seconds", "track_id", "class_name",
                "confidence", "x1", "y1", "x2", "y2",
            ])

            while frames_processed <= MAX_FRAMES:
                ok, frame = cap.read()

                if not ok:
                    break

                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=device,
                    classes=vehicle_ids,
                    conf=0.25,
                    imgsz=640,
                    save=False,
                    verbose=False,
                )[0]

                writer.write(result.plot(color_mode="instance"))

                # Nominal video timestamp, assuming constant FPS.
                timestamp = frames_processed / fps
                
                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    boxes = boxes.cpu()

                    coordinates = boxes.xyxy.tolist()
                    track_ids = boxes.id.int().tolist()
                    confidences = boxes.conf.tolist()
                    class_ids = boxes.cls.int().tolist()

                    for xyxy, track_id, confidence, class_id in zip(
                        coordinates, track_ids, confidences, class_ids
                    ):
                        x1, y1, x2, y2 = xyxy
                        seen_track_ids.add(track_id)

                        csv_writer.writerow([
                            frames_processed,
                            round(timestamp, 3),
                            track_id,
                            model.names[class_id],
                            round(confidence, 4),
                            round(x1, 2), round(y1, 2),
                            round(x2, 2), round(y2, 2),
                        ])

                frames_processed += 1
                if frames_processed % 50 == 0:
                    print(f"Processed {frames_processed} frames")

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Tracked observation rows: {detections_logged}")
    print(f"Distinct track IDs observed: {len(seen_track_ids)}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()