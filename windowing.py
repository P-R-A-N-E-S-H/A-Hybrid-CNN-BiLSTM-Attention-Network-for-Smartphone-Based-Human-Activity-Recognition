"""
windowing.py

Window generation utilities for the SisFall preprocessing pipeline.
"""

from collections import Counter

import numpy as np


class WindowGenerator:

    def __init__(self, sensor_columns):
        self.sensor_columns = sensor_columns

    def subject_windows(
            self,
            df,
            subject_column="subject",
            label_column="activity",
            window_size=200,
            overlap=100
    ):

        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")

        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be non-negative and smaller than window_size")

        step = window_size - overlap
        feature_count = len(self.sensor_columns)

        windows = []
        labels = []

        group_cols = [subject_column, label_column]

        for _, group in df.groupby(group_cols, sort=False):
            data = group[self.sensor_columns].to_numpy()
            activity_labels = group[label_column].to_numpy()

            if len(data) < window_size:
                continue

            for start in range(0, len(data) - window_size + 1, step):
                end = start + window_size
                window = data[start:end]
                label_window = activity_labels[start:end]

                if window.shape[0] != window_size:
                    continue

                if np.all(label_window == label_window[0]):
                    label = label_window[0]
                else:
                    label = Counter(label_window).most_common(1)[0][0]

                windows.append(window)
                labels.append(label)

        if len(windows) == 0:
            return np.empty((0, window_size, feature_count)), np.empty((0,), dtype=object)

        return np.stack(windows), np.asarray(labels)
