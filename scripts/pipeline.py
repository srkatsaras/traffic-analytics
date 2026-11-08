import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from .line_counter import LineCounter

# Define ROOT const and input/output directories
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/raw/vehicles.mp4"
OUTPUT_DIR = ROOT / "outputs/vehicle_counting"

MAX_FRAMES = 300

detections_logged = 0
seen_track_ids = set()

def process_video(
    input_path,
    output_dir,
    *,
    x_start=0.10,
    x_end=0.90,
    line_y=0.60,
    max_frames=None,
    on_progress=None,
):
    INPUT = Path(input_path)
    OUTPUT_DIR = Path(output_dir)

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

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frames = total_frames

        if max_frames is not None and total_frames > 0:
            expected_frames = min(total_frames, max_frames)

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

            counter = None
            crossing_events = []
            while max_frames is None or frames_processed < max_frames:
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

                    counter = LineCounter(
                        x1=int(width * x_start),
                        x2=int(width * x_end),
                        y=int(height * line_y),
                        dead_band=max(4, int(height * 0.01)),
                        max_gap=max(1, round(fps * 0.5)),
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

                        anchor_x = (x1 + x2) / 2
                        anchor_y = y2

                        direction = counter.update(
                            track_id,
                            anchor_x,
                            anchor_y,
                            frames_processed,
                        )

                        if direction is not None:
                            crossing_events.append({
                                "frame_index": frames_processed,
                                "time_seconds": round(timestamp, 3),
                                "track_id": track_id,
                                "direction": direction,
                                "class_name": model.names[class_id],
                            })

                        csv_writer.writerow([
                            frames_processed,
                            round(timestamp, 3),
                            track_id,
                            model.names[class_id],
                            round(confidence, 4),
                            round(x1, 2), round(y1, 2),
                            round(x2, 2), round(y2, 2),
                        ])

                annotated = result.plot(color_mode="instance")

                cv2.line(
                    annotated,
                    (counter.x1, counter.y),
                    (counter.x2, counter.y),
                    (0, 255, 255),
                    3,
                )

                label = (
                    f"UP: {counter.counts['up']}  "
                    f"DOWN: {counter.counts['down']}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

                writer.write(annotated)

                frames_processed += 1
                if (
                    on_progress is not None
                    and expected_frames > 0
                    and frames_processed % 50 == 0
                ):
                    on_progress(min(frames_processed / expected_frames, 0.99))

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Tracked observation rows: {detections_logged}")
    print(f"Distinct track IDs observed: {len(seen_track_ids)}")
    print(f"Outputs: {OUTPUT_DIR}")

    events_path = OUTPUT_DIR / "crossings.csv"
    with events_path.open("w", newline="") as file:
        fields = [
            "frame_index", "time_seconds", "track_id",
            "direction", "class_name",
        ]
        event_writer = csv.DictWriter(file, fieldnames=fields)
        event_writer.writeheader()
        event_writer.writerows(crossing_events)

    return {
        "frames_processed": frames_processed,
        "counts": dict(counter.counts),
        "artifacts": {
            "video": "annotated.mp4",
            "detections": "detections.csv",
            "crossings": "crossings.csv",
        },
    }
