import cv2
import threading

from detection.yolo_detector import YOLODetector
from tracking.deep_sort.tracker import DeepSORT

# ONE shared detector
detector = YOLODetector()


def camera_worker(source, window_name):
    cap = cv2.VideoCapture(source)
    tracker = DeepSORT()

    if not cap.isOpened():
        print(f"❌ Cannot open {source}")
        return

    print(f"✅ Stream started: {window_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        for t in tracks:
            x1, y1, x2, y2, tid = map(int, t)

            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(frame, f"ID {tid}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()


def main():
    streams = [
        0,                          # webcam
        "data/videos/people.mp4"    # video file
    ]

    threads = []

    for i, src in enumerate(streams):
        t = threading.Thread(
            target=camera_worker,
            args=(src, f"Camera {i}")
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
