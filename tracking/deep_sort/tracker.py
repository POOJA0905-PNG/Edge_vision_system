import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.missed = 0

class DeepSORT:
    def __init__(self):
        self.tracks = []
        self.next_id = 0

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB-xA) * max(0, yB-yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, detections):
        updated = []

        for det in detections:
            x1,y1,x2,y2,_,_ = det
            bbox = [x1,y1,x2,y2]

            matched = False
            for track in self.tracks:
                if self.iou(track.bbox, bbox) > 0.4:
                    track.bbox = bbox
                    track.missed = 0
                    updated.append(track)
                    matched = True
                    break

            if not matched:
                self.tracks.append(Track(bbox, self.next_id))
                self.next_id += 1

        results = []
        for track in self.tracks:
            x1,y1,x2,y2 = track.bbox
            results.append([x1,y1,x2,y2,track.id])

        return results
