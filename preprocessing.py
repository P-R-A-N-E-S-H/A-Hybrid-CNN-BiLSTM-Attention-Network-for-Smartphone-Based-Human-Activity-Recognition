"""
preprocessing.py

Human Activity Recognition Using Smartphones
UCI HAR Dataset - Raw Inertial Signals

Input:
    X_train -> (7352, 128, 9)
    X_test  -> (2947, 128, 9)

Output:
    X_train -> (5881, 128, 9)
    X_val   -> (1471, 128, 9)
    X_test  -> (2947, 128, 9)

Author : Pranesh

Project:
Human Activity Recognition using Deep Learning
"""

import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import HARRawDataLoader


class HARPreprocessor:

    def __init__(self):

        self.base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.processed_dir = os.path.join(
            self.base_dir,
            "data",
            "processed"
        )

        self.models_dir = os.path.join(
            self.base_dir,
            "models"
        )

        os.makedirs(
            self.processed_dir,
            exist_ok=True
        )

        os.makedirs(
            self.models_dir,
            exist_ok=True
        )

        # One scaler for the 9 sensor channels
        self.scaler = StandardScaler()

        self.activity_names = [
            "LAYING",
            "SITTING",
            "STANDING",
            "WALKING",
            "WALKING_DOWNSTAIRS",
            "WALKING_UPSTAIRS"
        ]

    ###########################################################
    # Dataset Path
    ###########################################################

    def get_dataset_path(self):

        return os.path.join(
            self.base_dir,
            "data",
            "human+activity+recognition+using+smartphones",
            "UCI HAR Dataset"
        )

    ###########################################################
    # Load Raw Dataset
    ###########################################################

    def load_raw_dataset(self):

        print()
        print("=" * 65)
        print("LOADING RAW UCI HAR DATASET")
        print("=" * 65)

        dataset_path = self.get_dataset_path()

        loader = HARRawDataLoader(
            dataset_path
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
            subject_train,
            subject_test
        ) = loader.load_dataset()

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            subject_train,
            subject_test
        )

    ###########################################################
    # Create Validation Set
    ###########################################################

    def create_validation_set(
        self,
        X_train,
        y_train,
        subject_train
    ):

        print()
        print("=" * 65)
        print("CREATING VALIDATION DATASET")
        print("=" * 65)

        (
            X_train,
            X_val,
            y_train,
            y_val,
            subject_train,
            subject_val
        ) = train_test_split(

            X_train,
            y_train,
            subject_train,

            test_size=0.20,

            random_state=42,

            stratify=y_train
        )

        print()
        print(
            "Training Samples   :",
            X_train.shape[0]
        )

        print(
            "Validation Samples :",
            X_val.shape[0]
        )

        return (
            X_train,
            X_val,
            y_train,
            y_val,
            subject_train,
            subject_val
        )

    ###########################################################
    # Normalize Sensor Data
    ###########################################################

    def normalize_data(
        self,
        X_train,
        X_val,
        X_test
    ):

        print()
        print("=" * 65)
        print("NORMALIZING SENSOR DATA")
        print("=" * 65)

        print()
        print("Before Normalization")

        print(
            "Train:",
            X_train.shape
        )

        print(
            "Validation:",
            X_val.shape
        )

        print(
            "Test:",
            X_test.shape
        )

        #######################################################
        # Reshape
        #
        # Original:
        # samples × 128 × 9
        #
        # Temporary:
        # (samples*128) × 9
        #######################################################

        train_2d = X_train.reshape(
            -1,
            X_train.shape[-1]
        )

        val_2d = X_val.reshape(
            -1,
            X_val.shape[-1]
        )

        test_2d = X_test.reshape(
            -1,
            X_test.shape[-1]
        )

        #######################################################
        # Fit ONLY on training data
        #######################################################

        train_2d = self.scaler.fit_transform(
            train_2d
        )

        #######################################################
        # Transform validation and test
        #######################################################

        val_2d = self.scaler.transform(
            val_2d
        )

        test_2d = self.scaler.transform(
            test_2d
        )

        #######################################################
        # Reshape Back
        #######################################################

        X_train = train_2d.reshape(
            X_train.shape
        )

        X_val = val_2d.reshape(
            X_val.shape
        )

        X_test = test_2d.reshape(
            X_test.shape
        )

        print()
        print("Normalization Completed")

        return (
            X_train,
            X_val,
            X_test
        )

    ###########################################################
    # Save Scaler
    ###########################################################

    def save_scaler(self):

        scaler_path = os.path.join(
            self.models_dir,
            "har_scaler.pkl"
        )

        joblib.dump(
            self.scaler,
            scaler_path
        )

        print()
        print("Scaler Saved")
        print(scaler_path)

    ###########################################################
    # Save Dataset
    ###########################################################

    def save_dataset(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ):

        print()
        print("=" * 65)
        print("SAVING PROCESSED DATASET")
        print("=" * 65)

        np.save(
            os.path.join(
                self.processed_dir,
                "X_train.npy"
            ),
            X_train.astype(
                np.float32
            )
        )

        np.save(
            os.path.join(
                self.processed_dir,
                "X_val.npy"
            ),
            X_val.astype(
                np.float32
            )
        )

        np.save(
            os.path.join(
                self.processed_dir,
                "X_test.npy"
            ),
            X_test.astype(
                np.float32
            )
        )

        np.save(
            os.path.join(
                self.processed_dir,
                "y_train.npy"
            ),
            y_train.astype(
                np.int64
            )
        )

        np.save(
            os.path.join(
                self.processed_dir,
                "y_val.npy"
            ),
            y_val.astype(
                np.int64
            )
        )

        np.save(
            os.path.join(
                self.processed_dir,
                "y_test.npy"
            ),
            y_test.astype(
                np.int64
            )
        )

        # Save class names
        class_names_path = os.path.join(
            self.processed_dir,
            "class_names.npy"
        )

        np.save(
            class_names_path,
            np.array(
                self.activity_names
            )
        )

        print()
        print("Saved:")

        print(
            os.path.join(
                self.processed_dir,
                "X_train.npy"
            )
        )

        print(
            os.path.join(
                self.processed_dir,
                "X_val.npy"
            )
        )

        print(
            os.path.join(
                self.processed_dir,
                "X_test.npy"
            )
        )

        print(
            os.path.join(
                self.processed_dir,
                "y_train.npy"
            )
        )

        print(
            os.path.join(
                self.processed_dir,
                "y_val.npy"
            )
        )

        print(
            os.path.join(
                self.processed_dir,
                "y_test.npy"
            )
        )

        print(
            class_names_path
        )

    ###########################################################
    # Validate Processed Dataset
    ###########################################################

    def validate_processed_data(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ):

        print()
        print("=" * 65)
        print("VALIDATING PROCESSED DATASET")
        print("=" * 65)

        #######################################################
        # Shape Checks
        #######################################################

        expected_train_shape = (
            5881,
            128,
            9
        )

        expected_val_shape = (
            1471,
            128,
            9
        )

        expected_test_shape = (
            2947,
            128,
            9
        )

        if X_train.shape != expected_train_shape:

            raise ValueError(
                f"Unexpected X_train shape: "
                f"{X_train.shape}"
            )

        if X_val.shape != expected_val_shape:

            raise ValueError(
                f"Unexpected X_val shape: "
                f"{X_val.shape}"
            )

        if X_test.shape != expected_test_shape:

            raise ValueError(
                f"Unexpected X_test shape: "
                f"{X_test.shape}"
            )

        #######################################################
        # Label Checks
        #######################################################

        if len(y_train) != len(X_train):

            raise ValueError(
                "Training X/y size mismatch."
            )

        if len(y_val) != len(X_val):

            raise ValueError(
                "Validation X/y size mismatch."
            )

        if len(y_test) != len(X_test):

            raise ValueError(
                "Testing X/y size mismatch."
            )

        #######################################################
        # NaN Checks
        #######################################################

        if np.isnan(X_train).any():

            raise ValueError(
                "NaN values found in X_train."
            )

        if np.isnan(X_val).any():

            raise ValueError(
                "NaN values found in X_val."
            )

        if np.isnan(X_test).any():

            raise ValueError(
                "NaN values found in X_test."
            )

        #######################################################
        # Label Range
        #######################################################

        if np.min(y_train) < 0 or np.max(y_train) > 5:

            raise ValueError(
                "Invalid training labels."
            )

        if np.min(y_val) < 0 or np.max(y_val) > 5:

            raise ValueError(
                "Invalid validation labels."
            )

        if np.min(y_test) < 0 or np.max(y_test) > 5:

            raise ValueError(
                "Invalid testing labels."
            )

        print()
        print("Shape Check       : PASS")
        print("NaN Check         : PASS")
        print("Label Check       : PASS")
        print("Dataset Integrity : PASS")

    ###########################################################
    # Print Statistics
    ###########################################################

    def print_statistics(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ):

        print()
        print("=" * 65)
        print("FINAL DATASET")
        print("=" * 65)

        print()
        print("X_train :", X_train.shape)
        print("X_val   :", X_val.shape)
        print("X_test  :", X_test.shape)

        print()

        print("y_train :", y_train.shape)
        print("y_val   :", y_val.shape)
        print("y_test  :", y_test.shape)

        print()

        print(
            "Number of Classes :",
            len(
                np.unique(y_train)
            )
        )

        print()

        print("Training Distribution")

        unique, counts = np.unique(
            y_train,
            return_counts=True
        )

        for label, count in zip(
            unique,
            counts
        ):

            print(
                f"{self.activity_names[label]:<25} : {count}"
            )

        print()

        print("Validation Distribution")

        unique, counts = np.unique(
            y_val,
            return_counts=True
        )

        for label, count in zip(
            unique,
            counts
        ):

            print(
                f"{self.activity_names[label]:<25} : {count}"
            )

        print()

        print("Testing Distribution")

        unique, counts = np.unique(
            y_test,
            return_counts=True
        )

        for label, count in zip(
            unique,
            counts
        ):

            print(
                f"{self.activity_names[label]:<25} : {count}"
            )

    ###########################################################
    # Complete Pipeline
    ###########################################################

    def run_pipeline(self):

        #######################################################
        # Load
        #######################################################

        (
            X_train,
            X_test,
            y_train,
            y_test,
            subject_train,
            subject_test
        ) = self.load_raw_dataset()

        #######################################################
        # Validation Split
        #######################################################

        (
            X_train,
            X_val,
            y_train,
            y_val,
            subject_train,
            subject_val
        ) = self.create_validation_set(
            X_train,
            y_train,
            subject_train
        )

        #######################################################
        # Normalize
        #######################################################

        (
            X_train,
            X_val,
            X_test
        ) = self.normalize_data(
            X_train,
            X_val,
            X_test
        )

        #######################################################
        # Save Scaler
        #######################################################

        self.save_scaler()

        #######################################################
        # Validate
        #######################################################

        self.validate_processed_data(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

        #######################################################
        # Save
        #######################################################

        self.save_dataset(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

        #######################################################
        # Statistics
        #######################################################

        self.print_statistics(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

        print()
        print("=" * 65)
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 65)

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )


############################################################
# Main
############################################################

def main():

    processor = HARPreprocessor()

    processor.run_pipeline()


if __name__ == "__main__":

    main()