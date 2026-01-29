import cv2

# detection
from detection.yolo_detector import YOLODetector

# segmentation
from segmentation.segmenter import Segmenter

# tracking (PURE PYTHON – no CMake)
from tracking.deep_sort.tracker import DeepSORT

# analytics
from analytics.counting import count_objects

# clustering
from clustering.anomaly import AnomalyDetector


def main():
    # -----------------------------
    # Initialize modules
    # -----------------------------
    detector = YOLODetector()
    segmenter = Segmenter()
    tracker = DeepSORT()
    anomaly = AnomalyDetector()

    cap = cv2.VideoCapture(0)   # webcam

    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    print("✅ Edge Vision pipeline started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --------------------------------
        # 1. Object Detection
        # --------------------------------
        detections = detector.detect(frame)

        # --------------------------------
        # 2. Tracking
        # --------------------------------
        tracks = tracker.update(detections)

        centroids = []

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            centroids.append([cx, cy])

            # bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ID
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # centroid
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # --------------------------------
        # 3. Counting
        # --------------------------------
        total_count = count_objects(tracks)

        cv2.putText(
            frame,
            f"Count: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # --------------------------------
        # 4. Clustering / Anomaly
        # --------------------------------
        labels = anomaly.detect(centroids)

        if len(labels) > 0:
            for i, label in enumerate(labels):
                if label == -1:   # anomaly
                    x, y = centroids[i]
                    cv2.circle(frame, (x, y), 10, (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        "ANOMALY",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

        # --------------------------------
        # Display
        # --------------------------------
        cv2.imshow("Edge Vision System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
