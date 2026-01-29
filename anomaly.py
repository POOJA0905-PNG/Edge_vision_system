from sklearn.cluster import DBSCAN
import numpy as np

class AnomalyDetector:
    def detect(self, centroids):
        if len(centroids) < 5:
            return []

        X = np.array(centroids)
        db = DBSCAN(eps=40, min_samples=3).fit(X)
        return db.labels_
