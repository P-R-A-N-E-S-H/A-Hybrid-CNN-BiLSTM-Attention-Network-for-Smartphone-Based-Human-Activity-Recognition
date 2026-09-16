import os
import pandas as pd

from inference import FallDetector


def test_resolve_resource_path_uses_module_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    resolved = FallDetector.resolve_path("models/cnn_lstm_quick_test.keras")

    assert os.path.isabs(resolved)
    assert os.path.exists(resolved)


def test_create_windows_handles_empty_dataframe():
    empty_df = pd.DataFrame(columns=FallDetector.sensor_columns())

    detector = FallDetector(
        model_path=os.path.join("models", "cnn_lstm_quick_test.keras"),
        scaler_path=os.path.join("models", "scaler.pkl"),
        encoder_path=os.path.join("models", "label_encoder.pkl"),
    )
    windows = detector.create_windows(empty_df, window_size=20, overlap=10)

    assert windows.shape == (0, 20, 6)
