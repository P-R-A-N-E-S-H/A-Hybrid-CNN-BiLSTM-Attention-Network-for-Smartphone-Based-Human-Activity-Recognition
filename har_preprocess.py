import os
import zipfile
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

HAR_SIGNALS = [
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

HAR_CLASSES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


class HARPreprocessor:

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        self.base_dir = base_dir
        self.dataset_dir = os.path.join(
            self.base_dir,
            "data",
            "human+activity+recognition+using+smartphones"
        )
        self.processed_dir = os.path.join(
            self.base_dir,
            "data",
            "processed",
            "har"
        )
        os.makedirs(self.processed_dir, exist_ok=True)

    def extract_zip(self, archive_name="UCI HAR Dataset.zip"):
        extract_path = os.path.join(self.dataset_dir, "UCI HAR Dataset")
        if os.path.isdir(extract_path):
            print(f"HAR dataset already extracted: {extract_path}")
            return extract_path

        archive_path = os.path.join(self.dataset_dir, archive_name)
        if not os.path.exists(archive_path):
            raise FileNotFoundError(
                f"HAR archive not found: {archive_path}.\n"
                f"Place the zip archive under {self.dataset_dir} or extract the dataset into {extract_path}."
            )

        print("Extracting HAR zip archive...")
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(self.dataset_dir)

        print(f"Extracted HAR dataset to {extract_path}")
        return extract_path

    def load_signals(self, har_root, subset):
        signals = []
        for signal_name in HAR_SIGNALS:
            filepath = os.path.join(
                har_root,
                subset,
                "Inertial Signals",
                f"{signal_name}_{subset}.txt"
            )
            if not os.path.exists(filepath):
                raise FileNotFoundError(filepath)

            signals.append(np.loadtxt(filepath))

        # Shape: (samples, timesteps, features)
        return np.stack(signals, axis=-1)

    def load_labels(self, har_root, subset):
        filepath = os.path.join(har_root, subset, f"y_{subset}.txt")
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        y = np.loadtxt(filepath, dtype=int).astype(int)
        return y - 1

    def add_magnitude_features(self, X):
        total_acc = X[..., 0:3]
        body_acc = X[..., 3:6]
        body_gyro = X[..., 6:9]

        total_acc_mag = np.linalg.norm(total_acc, axis=-1, keepdims=True)
        body_acc_mag = np.linalg.norm(body_acc, axis=-1, keepdims=True)
        body_gyro_mag = np.linalg.norm(body_gyro, axis=-1, keepdims=True)

        return np.concatenate(
            [X, total_acc_mag, body_acc_mag, body_gyro_mag],
            axis=-1
        )

    def load_har_dataset(self):
        har_root = self.extract_zip()

        print("Loading HAR train signals...")
        X_train = self.load_signals(har_root, "train")
        y_train = self.load_labels(har_root, "train")

        print("Loading HAR test signals...")
        X_test = self.load_signals(har_root, "test")
        y_test = self.load_labels(har_root, "test")

        X_train = self.add_magnitude_features(X_train)
        X_test = self.add_magnitude_features(X_test)

        print("HAR data shapes:")
        print("X_train", X_train.shape)
        print("y_train", y_train.shape)
        print("X_test", X_test.shape)
        print("y_test", y_test.shape)

        return X_train, y_train, X_test, y_test

    def normalize_features(self, X_train, X_val, X_test):
        n_train, timesteps, features = X_train.shape
        train_flat = X_train.reshape(-1, features)

        print("Fitting scaler on training features...")
        scaler = StandardScaler()
        scaler.fit(train_flat)

        def transform_array(X):
            shape = X.shape
            flat = X.reshape(-1, shape[-1])
            scaled = scaler.transform(flat)
            return scaled.reshape(shape)

        return scaler, transform_array(X_train), transform_array(X_val), transform_array(X_test)

    def build_label_encoder(self):
        encoder = LabelEncoder()
        encoder.fit(HAR_CLASSES)
        return encoder

    def save_processed_data(
            self,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            scaler,
            label_encoder
    ):
        print("Saving processed HAR dataset...")
        os.makedirs(self.processed_dir, exist_ok=True)

        np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.processed_dir, "X_val.npy"), X_val)
        np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.processed_dir, "y_val.npy"), y_val)
        np.save(os.path.join(self.processed_dir, "y_test.npy"), y_test)

        joblib.dump(scaler, os.path.join(self.processed_dir, "scaler.pkl"))
        joblib.dump(label_encoder, os.path.join(self.processed_dir, "label_encoder.pkl"))

        with open(os.path.join(self.processed_dir, "classes.json"), "w") as f:
            json.dump(HAR_CLASSES, f, indent=4)

        print(f"Saved processed data to {self.processed_dir}")

    def split_train_validation(self, X_train, y_train, validation_size=0.15, random_state=42):
        print("Creating HAR validation split...")
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            X_train,
            y_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train
        )
        print("Split shapes:")
        print("X_train", X_train_new.shape)
        print("X_val", X_val.shape)
        print("y_train", y_train_new.shape)
        print("y_val", y_val.shape)
        return X_train_new, X_val, y_train_new, y_val

    def run(self, validation_size=0.15, random_state=42):
        X_train, y_train, X_test, y_test = self.load_har_dataset()
        X_train, X_val, y_train, y_val = self.split_train_validation(
            X_train,
            y_train,
            validation_size=validation_size,
            random_state=random_state
        )

        scaler, X_train, X_val, X_test = self.normalize_features(
            X_train,
            X_val,
            X_test
        )

        label_encoder = self.build_label_encoder()
        self.save_processed_data(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            scaler,
            label_encoder
        )

        return {
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape,
            "classes": HAR_CLASSES
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess UCI HAR dataset")
    parser.add_argument("--validation_size", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    preprocessor = HARPreprocessor()
    stats = preprocessor.run(
        validation_size=args.validation_size,
        random_state=args.random_state
    )
    print(json.dumps(stats, indent=2))
