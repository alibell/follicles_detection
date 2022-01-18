import numpy as np
import random


class ObjectDetector:
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        predicted_locations = []
        for image_path in X:
            # image = load_image(image_path)
            # etc.
            predictions_for_this_image = [
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (1084, 5189, 1218, 5423),
                },
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (2564, 3543, 2676, 3754),
                },
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (3224, 8979, 3417, 9092),
                },
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (7541, 353, 7765, 505),
                },
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (7792, 10523, 7971, 10601),
                },
                {
                    "proba": random.random(),
                    "class": "Primordial",
                    "bbox": (11807, 378, 11951, 483),
                },
                {
                    "proba": random.random(),
                    "class": "Secondary",
                    "bbox": (9569, 503, 10830, 1850),
                },
                {
                    "proba": random.random(),
                    "class": "Tertiary",
                    "bbox": (2943, 1791, 4973, 3683),
                },
                {
                    "proba": random.random(),
                    "class": "Tertiary",
                    "bbox": (7388, 8841, 9232, 10492),
                },
                {
                    "proba": random.random(),
                    "class": "Tertiary",
                    "bbox": (13188, 526, 15346, 2573),
                },
            ]
            predicted_locations.append(predictions_for_this_image)

        y_pred = np.empty(len(X), dtype=object)
        y_pred[:] = predicted_locations
        return y_pred