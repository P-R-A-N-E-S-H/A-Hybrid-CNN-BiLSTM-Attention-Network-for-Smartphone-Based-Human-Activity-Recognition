"""
data_loader.py

Human Activity Recognition Using Smartphones
UCI HAR Dataset - Raw Inertial Signals

Author : Pranesh

Project:
Human Activity Recognition using Deep Learning
"""

import os
import numpy as np


class HARRawDataLoader:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.train_path = os.path.join(
            dataset_path,
            "train"
        )

        self.test_path = os.path.join(
            dataset_path,
            "test"
        )

        self.train_signal_path = os.path.join(
            self.train_path,
            "Inertial Signals"
        )

        self.test_signal_path = os.path.join(
            self.test_path,
            "Inertial Signals"
        )

    ###########################################################
    # Check Dataset
    ###########################################################

    def check_dataset(self):

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        if not os.path.exists(self.train_signal_path):
            raise FileNotFoundError(
                f"Training Inertial Signals folder not found:\n"
                f"{self.train_signal_path}"
            )

        if not os.path.exists(self.test_signal_path):
            raise FileNotFoundError(
                f"Testing Inertial Signals folder not found:\n"
                f"{self.test_signal_path}"
            )

        print("=" * 65)
        print("UCI HAR RAW DATASET FOUND")
        print("=" * 65)
        print(self.dataset_path)

    ###########################################################
    # Signal Names
    ###########################################################

    def get_signal_names(self):

        return [
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",

            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",

            "total_acc_x",
            "total_acc_y",
            "total_acc_z"
        ]

    ###########################################################
    # Load One Signal
    ###########################################################

    def load_signal(
        self,
        folder,
        signal_name,
        split
    ):

        filename = os.path.join(
            folder,
            f"{signal_name}_{split}.txt"
        )

        if not os.path.exists(filename):

            raise FileNotFoundError(
                f"Signal file not found:\n{filename}"
            )

        data = np.loadtxt(filename)

        return data

    ###########################################################
    # Load All 9 Signals
    ###########################################################

    def load_signals(
        self,
        folder,
        split
    ):

        signal_names = self.get_signal_names()

        signals = []

        print()
        print("Loading Signals")
        print("-" * 65)

        for signal_name in signal_names:

            data = self.load_signal(
                folder,
                signal_name,
                split
            )

            print(
                f"{signal_name:<20} : {data.shape}"
            )

            signals.append(data)

        return signals

    ###########################################################
    # Create 3D Tensor
    ###########################################################

    def create_tensor(
        self,
        folder,
        split
    ):

        signals = self.load_signals(
            folder,
            split
        )

        # Each signal:
        #
        # (samples, 128)
        #
        # Stack along last axis:
        #
        # (samples, 128, 9)

        tensor = np.stack(
            signals,
            axis=-1
        )

        return tensor

    ###########################################################
    # Load Labels
    ###########################################################

    def load_labels(self, split):

        filename = os.path.join(
            self.dataset_path,
            split,
            f"y_{split}.txt"
        )

        if not os.path.exists(filename):

            raise FileNotFoundError(
                f"Label file not found:\n{filename}"
            )

        labels = np.loadtxt(
            filename,
            dtype=np.int64
        )

        # Original labels:
        # 1, 2, 3, 4, 5, 6
        #
        # Convert to:
        # 0, 1, 2, 3, 4, 5

        labels = labels - 1

        return labels

    ###########################################################
    # Load Subjects
    ###########################################################

    def load_subjects(self, split):

        filename = os.path.join(
            self.dataset_path,
            split,
            f"subject_{split}.txt"
        )

        if not os.path.exists(filename):

            raise FileNotFoundError(
                f"Subject file not found:\n{filename}"
            )

        subjects = np.loadtxt(
            filename,
            dtype=np.int64
        )

        return subjects

    ###########################################################
    # Activity Names
    ###########################################################

    def get_activity_names(self):

        return [
            "LAYING",
            "SITTING",
            "STANDING",
            "WALKING",
            "WALKING_DOWNSTAIRS",
            "WALKING_UPSTAIRS"
        ]

    ###########################################################
    # Load Complete Dataset
    ###########################################################

    def load_dataset(self):

        self.check_dataset()

        print()
        print("=" * 65)
        print("LOADING RAW INERTIAL SIGNALS")
        print("=" * 65)

        #######################################################
        # TRAINING DATA
        #######################################################

        print()
        print("TRAINING DATA")
        print("=" * 65)

        X_train = self.create_tensor(
            self.train_signal_path,
            "train"
        )

        y_train = self.load_labels(
            "train"
        )

        subject_train = self.load_subjects(
            "train"
        )

        #######################################################
        # TESTING DATA
        #######################################################

        print()
        print("TESTING DATA")
        print("=" * 65)

        X_test = self.create_tensor(
            self.test_signal_path,
            "test"
        )

        y_test = self.load_labels(
            "test"
        )

        subject_test = self.load_subjects(
            "test"
        )

        #######################################################
        # Validation
        #######################################################

        self.validate_data(
            X_train,
            y_train,
            subject_train,
            "Training"
        )

        self.validate_data(
            X_test,
            y_test,
            subject_test,
            "Testing"
        )

        #######################################################
        # Final Information
        #######################################################

        print()
        print("=" * 65)
        print("RAW DATASET LOADED SUCCESSFULLY")
        print("=" * 65)

        print()
        print(
            "X_train :",
            X_train.shape
        )

        print(
            "X_test  :",
            X_test.shape
        )

        print(
            "y_train :",
            y_train.shape
        )

        print(
            "y_test  :",
            y_test.shape
        )

        print()
        print(
            "Input Shape :",
            X_train.shape[1:]
        )

        print(
            "Number of Classes :",
            len(self.get_activity_names())
        )

        print()
        print("Activities")

        for index, activity in enumerate(
            self.get_activity_names()
        ):

            print(
                f"{index} -> {activity}"
            )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            subject_train,
            subject_test
        )

    ###########################################################
    # Validate Dataset
    ###########################################################

    def validate_data(
        self,
        X,
        y,
        subjects,
        name
    ):

        print()
        print(f"{name} Validation")
        print("-" * 65)

        print(
            "Samples    :",
            X.shape[0]
        )

        print(
            "Time Steps :",
            X.shape[1]
        )

        print(
            "Sensors    :",
            X.shape[2]
        )

        print(
            "Labels     :",
            y.shape[0]
        )

        print(
            "Subjects   :",
            subjects.shape[0]
        )

        if X.shape[0] != y.shape[0]:

            raise ValueError(
                f"{name}: X and y sample count mismatch."
            )

        if X.shape[0] != subjects.shape[0]:

            raise ValueError(
                f"{name}: X and subjects count mismatch."
            )

        if X.shape[1] != 128:

            raise ValueError(
                f"{name}: Expected 128 time steps, "
                f"got {X.shape[1]}."
            )

        if X.shape[2] != 9:

            raise ValueError(
                f"{name}: Expected 9 sensor channels, "
                f"got {X.shape[2]}."
            )

    ###########################################################
    # Dataset Summary
    ###########################################################

    def summary(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        activities = self.get_activity_names()

        print()
        print("=" * 65)
        print("DATASET SUMMARY")
        print("=" * 65)

        print()
        print(
            "Training Shape :",
            X_train.shape
        )

        print(
            "Testing Shape  :",
            X_test.shape
        )

        print()
        print("Training Class Distribution")

        unique, counts = np.unique(
            y_train,
            return_counts=True
        )

        for label, count in zip(
            unique,
            counts
        ):

            print(
                f"{activities[label]:<25} : {count}"
            )

        print()
        print("Testing Class Distribution")

        unique, counts = np.unique(
            y_test,
            return_counts=True
        )

        for label, count in zip(
            unique,
            counts
        ):

            print(
                f"{activities[label]:<25} : {count}"
            )


############################################################
# Main
############################################################

def main():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    dataset = os.path.join(
        base_dir,
        "data",
        "human+activity+recognition+using+smartphones",
        "UCI HAR Dataset"
    )

    loader = HARRawDataLoader(
        dataset
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        subject_train,
        subject_test
    ) = loader.load_dataset()

    loader.summary(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":

    main()