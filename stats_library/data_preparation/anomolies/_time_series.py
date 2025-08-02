import numpy as np

__all__ = [
    "MovingZScoreAnomalyDetector",
]

class MovingZScoreAnomalyDetector:
    def __init__(self, window_size=15, time_step=0.5, num_std=3):
        self.window_size = window_size
        self.time_step = time_step
        self.num_std = num_std

        self.anomaly_indices = []
        self.mean_values = None
        self.std_values = None

    def fit(self, y_data):
        y_data = np.asarray(y_data)
        self.anomaly_indices = []
        rolling_stats = []

        for i in range(len(y_data) - self.window_size):
            window_indices = [j for j in range(i, i + self.window_size) if j not in self.anomaly_indices]
            if not window_indices:
                rolling_stats.append([np.nan, np.nan])
                continue

            window_data = y_data[window_indices]
            mean_value = np.mean(window_data)
            std_value = np.std(window_data)
            rolling_stats.append([mean_value, std_value])

            current_value = y_data[i + self.window_size]
            if np.abs(current_value - mean_value) > self.num_std * std_value:
                self.anomaly_indices.append(i + self.window_size)

        self.mean_values, self.std_values = np.array(rolling_stats).T
        return self

    def get_anomalies(self):
        return self.anomaly_indices

    def get_stats(self):
        return self.mean_values, self.std_values
