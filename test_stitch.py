import cv2
from stitching.stitcher import CameraStitcher

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("data/videos/people.mp4")

stitcher = CameraStitcher()

TARGET_HEIGHT = 480

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()

    if not r1 or not r2:
        break

    # resize both frames to same height
    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]

    f1r = cv2.resize(f1, (int(w1 * TARGET_HEIGHT / h1), TARGET_HEIGHT))
    f2r = cv2.resize(f2, (int(w2 * TARGET_HEIGHT / h2), TARGET_HEIGHT))

    pano = stitcher.stitch(f1r, f2r)

    if pano is not None:
        cv2.imshow("Panoramic View", pano)
    else:
        combined = cv2.hconcat([f1r, f2r])
        cv2.imshow("Panoramic View (No Overlap)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
