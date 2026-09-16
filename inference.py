"""
inference.py

HAR Inference Module

Human Activity Recognition using UCI HAR Dataset

Author: Pranesh
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import models


class FallDetector:
    def __init__(self, model_path="models/cnn_lstm_improved.keras", scaler_path="data/processed/har/scaler.pkl", encoder_path="data/processed/har/label_encoder.pkl"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = self.resolve_path(model_path)
        scaler_path = self.resolve_path(scaler_path)
        encoder_path = self.resolve_path(encoder_path)

        print("=" * 60)
        print("Loading resources:")
        print(f" Model: {model_path}")
        print(f" Scaler: {scaler_path}")
        print(f" Encoder: {encoder_path}")
        print("=" * 60)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")

        from tensorflow.keras.layers import Layer

        @tf.keras.utils.register_keras_serializable(package='models')
        class Attention(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def build(self, input_shape):
                self.W = self.add_weight(
                    name='att_weight',
                    shape=(input_shape[-1], input_shape[-1]),
                    initializer='glorot_uniform',
                    trainable=True
                )
                self.b = self.add_weight(
                    name='att_bias',
                    shape=(input_shape[-1],),
                    initializer='zeros',
                    trainable=True
                )
                self.u = self.add_weight(
                    name='att_context',
                    shape=(input_shape[-1],),
                    initializer='glorot_uniform',
                    trainable=True
                )
                super().build(input_shape)

            def call(self, inputs):
                u = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
                score = tf.tensordot(u, self.u, axes=1)
                weights = tf.nn.softmax(score, axis=1)
                context = tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)
                return context

        try:
            models.Attention = Attention
        except Exception:
            pass

        try:
            self.model = load_model(model_path, custom_objects={'Attention': Attention})
        except Exception:
            self.model = load_model(model_path)

        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)

        print("Model Loaded")
        print("Scaler Loaded")
        print("Label Encoder Loaded")
        print("=" * 60)

    @staticmethod
    def resolve_path(path):
        if path is None:
            return None
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))

    @staticmethod
    def sensor_columns():
        return [
            "total_acc_x",
            "total_acc_y",
            "total_acc_z",
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
        ]

    @staticmethod
    def add_magnitude_features(X):
        total_acc = X[..., 0:3]
        body_acc = X[..., 3:6]
        body_gyro = X[..., 6:9]
        total_acc_mag = np.linalg.norm(total_acc, axis=-1, keepdims=True)
        body_acc_mag = np.linalg.norm(body_acc, axis=-1, keepdims=True)
        body_gyro_mag = np.linalg.norm(body_gyro, axis=-1, keepdims=True)
        return np.concatenate([X, total_acc_mag, body_acc_mag, body_gyro_mag], axis=-1)

    def normalize(self, X):
        samples, timesteps, features = X.shape
        X_flat = X.reshape(-1, features)
        if hasattr(self.scaler, "feature_names_in_"):
            try:
                df = pd.DataFrame(X_flat, columns=self.scaler.feature_names_in_)
                X_flat = self.scaler.transform(df)
            except Exception:
                X_flat = self.scaler.transform(X_flat)
        else:
            X_flat = self.scaler.transform(X_flat)
        return X_flat.reshape(samples, timesteps, features)

    def load_csv(self, csv_path):
        csv_path = self.resolve_path(csv_path)
        df = pd.read_csv(csv_path)
        print("CSV Loaded", df.shape)
        return df

    def create_windows(self, df, window_size=128, overlap=64):
        expected = self.sensor_columns()
        if all(col in df.columns for col in expected):
            working_df = df
        else:
            compact = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            if all(col in df.columns for col in compact):
                working_df = pd.DataFrame()
                working_df["total_acc_x"] = df["acc_x"]
                working_df["total_acc_y"] = df["acc_y"]
                working_df["total_acc_z"] = df["acc_z"]
                working_df["body_acc_x"] = df["acc_x"]
                working_df["body_acc_y"] = df["acc_y"]
                working_df["body_acc_z"] = df["acc_z"]
                working_df["body_gyro_x"] = df["gyro_x"]
                working_df["body_gyro_y"] = df["gyro_y"]
                working_df["body_gyro_z"] = df["gyro_z"]
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= len(expected):
                    working_df = df[numeric_cols[: len(expected)]].copy()
                    working_df.columns = expected
                else:
                    raise ValueError(f"Input CSV missing expected sensor columns. Expected {expected}, found {list(df.columns)}")

        sensor_data = working_df[expected].values
        step = window_size - overlap
        windows = []
        n = len(sensor_data)
        if n == 0:
            windows = np.empty((0, window_size, sensor_data.shape[1] if sensor_data.ndim > 1 else 6), dtype=float)
        elif n < window_size:
            last_row = sensor_data[-1]
            padding = np.repeat(last_row[np.newaxis, :], window_size - n, axis=0)
            windows.append(np.vstack([sensor_data, padding]))
            windows = np.array(windows)
        else:
            for start in range(0, n - window_size + 1, step):
                windows.append(sensor_data[start:start + window_size])
            windows = np.array(windows)

        print("Sliding Windows Created", windows.shape)
        return windows

    def predict(self, windows):
        if windows.shape[-1] == 9:
            windows = self.add_magnitude_features(windows)
        windows = self.normalize(windows)
        probabilities = self.model.predict(windows, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        labels = self.label_encoder.inverse_transform(y_pred)
        return labels, confidence, probabilities

    def print_predictions(self, labels, confidence):
        print("\n" + "=" * 70)
        print("Prediction Results")
        print("=" * 70)
        for i, label in enumerate(labels, start=1):
            print(f"Window {i:03d} | Activity : {label:<5} | Confidence : {confidence[i-1]*100:.2f}%")
        print("=" * 70)

    def final_prediction(self, labels):
        values, counts = np.unique(labels, return_counts=True)
        prediction = values[np.argmax(counts)]
        print("\n" + "=" * 60)
        print("Final Prediction")
        print("=" * 60)
        print(f"Predicted Activity : {prediction}")
        return prediction

    def run(self, csv_path):
        print("\n" + "=" * 70)
        print("Running Inference")
        print("=" * 70)
        df = self.load_csv(csv_path)
        windows = self.create_windows(df)
        labels, confidence, _ = self.predict(windows)
        self.print_predictions(labels, confidence)
        self.final_prediction(labels)
        print("\nInference Completed Successfully")


def main():
    model_path = "models/cnn_lstm_improved.keras"
    scaler_path = "data/processed/har/scaler.pkl"
    encoder_path = "data/processed/har/label_encoder.pkl"
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_sample.csv")

    if not os.path.exists(test_file):
        try:
            X_test = np.load(os.path.join("data", "processed", "har", "X_test.npy"))
            sample = X_test[0]
            cols = FallDetector.sensor_columns()
            if sample.shape[1] >= len(cols):
                sample = sample[:, : len(cols)]
            pd.DataFrame(sample, columns=cols).to_csv(test_file, index=False)
            print(f"Wrote sample test CSV to {test_file}")
        except Exception as e:
            print("Failed to create sample CSV from processed HAR data:", e)
            print("Please provide a CSV with sensor columns or run har_preprocess.py")
            return

    detector = FallDetector(model_path=model_path, scaler_path=scaler_path, encoder_path=encoder_path)
    detector.run(test_file)


if __name__ == "__main__":
    main()
