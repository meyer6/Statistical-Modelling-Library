import numpy as np
import pytest
from stats_library.data_preparation.anomolies._time_series import MovingZScoreAnomalyDetector  # Replace with actual import path

@pytest.fixture
def sample_data():
    # Mostly normal data with a couple of injected anomalies
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=1.0, size=200)
    data[50] = 10    # Anomaly
    data[120] = -8   # Another anomaly
    return data

def test_detects_known_anomalies(sample_data):
    detector = MovingZScoreAnomalyDetector(window_size=15, num_std=3)
    detector.fit(sample_data)
    anomalies = detector.get_anomalies()

    assert 50 in anomalies, "Did not detect known positive anomaly at index 50"
    assert 120 in anomalies, "Did not detect known negative anomaly at index 120"

def test_no_false_positives_on_clean_data():
    np.random.seed(0)
    clean_data = np.random.uniform(0, 1, 100)
    detector = MovingZScoreAnomalyDetector(window_size=15, num_std=3)
    detector.fit(clean_data)
    anomalies = detector.get_anomalies()

    assert len(anomalies) == 0, f"Expected no anomalies, got {len(anomalies)}"

def test_output_shapes(sample_data):
    detector = MovingZScoreAnomalyDetector(window_size=10)
    detector.fit(sample_data)
    means, stds = detector.get_stats()

    expected_length = len(sample_data) - detector.window_size
    assert len(means) == expected_length
    assert len(stds) == expected_length

# def test_handles_all_anomalies_gracefully():
#     # Construct data where all points after the window are huge anomalies
#     data = np.concatenate([np.ones(20), np.full(20, 100)])
#     detector = MovingZScoreAnomalyDetector(window_size=10, num_std=2)
#     detector.fit(data)
#     anomalies = detector.get_anomalies()

#     # All of the second half should be detected as anomalies
#     assert set(anomalies) >= set(range(20, 40))
