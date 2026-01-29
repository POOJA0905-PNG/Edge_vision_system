import cv2
import numpy as np


class CameraStitcher:
    def __init__(self):
        self.orb = cv2.ORB_create(3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def stitch(self, img1, img2):
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return None

        matches = self.matcher.knnMatch(des1, des2, 2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 10:
            return None

        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return None

        h, w = img2.shape[:2]

        warped = cv2.warpPerspective(
            img1,
            H,
            (w * 2, h)
        )

        warped[0:h, 0:w] = img2
        return warped
