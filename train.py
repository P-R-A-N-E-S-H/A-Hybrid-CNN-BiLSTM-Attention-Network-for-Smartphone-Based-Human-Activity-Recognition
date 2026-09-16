"""
train.py

Training Module

AI-Powered Wearable Fall Detection
using SisFall Dataset

Author: Pranesh
"""

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import (

    EarlyStopping,

    ReduceLROnPlateau,

    ModelCheckpoint,

    CSVLogger

)

from tensorflow.keras.utils import plot_model

from models import DeepModels
from evaluate import ModelEvaluator


class Trainer:

    def __init__(self):

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.data_path = os.path.join(

            self.base_dir,

            "data",

            "processed",

            "har"

        )

        self.classes_path = os.path.join(

            self.data_path,

            "classes.json"

        )

        self.model_path = os.path.join(

            self.base_dir,

            "models"

        )

        self.result_path = os.path.join(

            self.base_dir,

            "results"

        )

        os.makedirs(

            self.model_path,

            exist_ok=True

        )

        os.makedirs(

            self.result_path,

            exist_ok=True

        )

    ############################################################
    # Load Dataset
    ############################################################

    def load_data(self):

        print("=" * 70)

        print("Loading Processed Dataset")

        print("=" * 70)

        if not os.path.exists(os.path.join(self.data_path, "X_train.npy")):
            print("Processed HAR data not found. Running HAR preprocessing...")
            from har_preprocess import HARPreprocessor
            HARPreprocessor(base_dir=self.base_dir).run()

        X_train = np.load(

            os.path.join(

                self.data_path,

                "X_train.npy"

            )

        )

        X_val = np.load(

            os.path.join(

                self.data_path,

                "X_val.npy"

            )

        )

        X_test = np.load(

            os.path.join(

                self.data_path,

                "X_test.npy"

            )

        )

        y_train = np.load(

            os.path.join(

                self.data_path,

                "y_train.npy"

            )

        )

        y_val = np.load(

            os.path.join(

                self.data_path,

                "y_val.npy"

            )

        )

        y_test = np.load(

            os.path.join(

                self.data_path,

                "y_test.npy"

            )

        )

        print()

        print("Training   :", X_train.shape)

        print("Validation :", X_val.shape)

        print("Testing    :", X_test.shape)

        return (

            X_train,

            X_val,

            X_test,

            y_train,

            y_val,

            y_test

        )

    ############################################################
    # Build Selected Model
    ############################################################

    def create_model(

            self,

            model_name,

            input_shape,

            num_classes,
            learning_rate=0.001

    ):

        builder = DeepModels(

            input_shape=input_shape,

            num_classes=num_classes,

            learning_rate=learning_rate

        )

        if model_name == "cnn":

            model = builder.build_cnn()

        elif model_name == "lstm":

            model = builder.build_lstm()

        elif model_name == "bilstm":

            model = builder.build_bilstm()

        elif model_name == "cnn_lstm":

            model = builder.build_cnn_lstm()

        elif model_name == "cnn_lstm_improved":

            model = builder.build_cnn_lstm_improved()

        else:

            raise ValueError(

                "Invalid model name"

            )

        return model

    ############################################################
    # Callbacks
    ############################################################

    def get_callbacks(
            self,
            model_name
    ):

        callback_list = [

            ####################################################
            # Early Stopping
            ####################################################

            EarlyStopping(

                monitor="val_loss",

                patience=10,

                restore_best_weights=True,

                verbose=1

            ),

            ####################################################
            # Reduce Learning Rate
            ####################################################

            ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.5,

                patience=5,

                min_lr=1e-6,

                verbose=1

            ),

            ####################################################
            # Save Best Model
            ####################################################

            ModelCheckpoint(

                filepath=os.path.join(

                    self.model_path,

                    f"{model_name}.keras"

                ),

                monitor="val_accuracy",

                save_best_only=True,

                save_weights_only=False,

                verbose=1

            ),

            ####################################################
            # Save Training Log
            ####################################################

            CSVLogger(

                os.path.join(

                    self.result_path,

                    f"{model_name}_training.csv"

                )

            )

        ]

        return callback_list

    ############################################################
    # Save Model Architecture
    ############################################################

    def save_architecture(

            self,

            model,

            model_name

    ):

        print("=" * 70)

        print("Saving Model Architecture")

        print("=" * 70)

        plot_model(

            model,

            to_file=os.path.join(

                self.result_path,

                f"{model_name}_architecture.png"

            ),

            show_shapes=True,

            show_dtype=True,

            show_layer_names=True,

            expand_nested=True,

            dpi=150

        )

        print(

            f"Architecture Saved : "

            f"{model_name}_architecture.png"

        )
    ############################################################
    # Train Model
    ############################################################

    def train_model(

            self,

            model,

            X_train,

            y_train,

            X_val,

            y_val,

            model_name,

            epochs=50,

            batch_size=64,

            class_weight=None,

            use_tf_data=True,

            quick_run=False,

            quick_samples=2048

    ):

        print("=" * 70)

        print(f"Training {model_name.upper()} Model")

        print("=" * 70)

        # Quick-run mode: use a small subset for fast iteration
        if quick_run:
            print(f"Quick run enabled - using first {quick_samples} samples for training")
            X_train = X_train[:quick_samples]
            y_train = y_train[:quick_samples]
            val_samples = min(max(256, quick_samples // 4), X_val.shape[0])
            X_val = X_val[:val_samples]
            y_val = y_val[:val_samples]

        # If using tf.data and no class weights, build tf.data pipelines
        if use_tf_data and class_weight is None:
            print("Using tf.data pipeline (cache, prefetch) for training data")

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            # Cast and parallel preprocess (no-op here but ready for preprocess_fn)
            train_ds = train_ds.shuffle(buffer_size=10000)
            train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=self.get_callbacks(model_name),
                verbose=1
            )
        else:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=self.get_callbacks(model_name),
                class_weight=class_weight,
                verbose=1,
                shuffle=True
            )

        print("\nTraining Completed Successfully")

        return history

    ############################################################
    # Evaluate Model
    ############################################################

    def evaluate_model(

            self,

            model,

            history,

            X_test,

            y_test,

            class_names=None

    ):

        print("=" * 70)

        print("Evaluating Model")

        print("=" * 70)

        evaluator = ModelEvaluator(

            output_dir=self.result_path

        )

        metrics = evaluator.evaluate_dl_model(

            model,

            X_test,

            y_test,

            class_names

        )

        evaluator.plot_training_history(

            history

        )

        return metrics

    ############################################################
    # Save Training History
    ############################################################

    def save_history(

            self,

            history,

            model_name

    ):

        history_file = os.path.join(

            self.result_path,

            f"{model_name}_history.npy"

        )

        np.save(

            history_file,

            history.history,

            allow_pickle=True

        )

        print()

        print(f"History Saved : {history_file}")
    ############################################################
    # Complete Training Pipeline
    ############################################################

    def run(

            self,

            model_name="cnn_lstm",

            epochs=50,

            batch_size=64,

            learning_rate=0.001,

            class_weight=None,

            class_names=None,

            use_tf_data=True,

            quick_run=False,

            quick_samples=2048

    ):

        ########################################################
        # Load Dataset
        ########################################################

        (

            X_train,

            X_val,

            X_test,

            y_train,

            y_val,

            y_test

        ) = self.load_data()

        ########################################################
        # Model Information
        ########################################################

        input_shape = (

            X_train.shape[1],

            X_train.shape[2]

        )

        num_classes = len(

            np.unique(y_train)

        )

        print("=" * 70)

        print("Training Configuration")

        print("=" * 70)

        print(f"Model Name     : {model_name}")

        print(f"Input Shape    : {input_shape}")

        print(f"Classes        : {num_classes}")

        print(f"Epochs         : {epochs}")

        print(f"Batch Size     : {batch_size}")

        ########################################################
        # Build Model
        ########################################################

        model = self.create_model(
            model_name,
            input_shape,
            num_classes,
            learning_rate=learning_rate
        )

        model.summary()

        ########################################################
        # Save Architecture
        ########################################################

        self.save_architecture(

            model,

            model_name

        )

        ########################################################
        # Train Model
        ########################################################

        history = self.train_model(

            model,

            X_train,

            y_train,

            X_val,

            y_val,

            model_name,

            epochs,

            batch_size,

            class_weight=class_weight,

            use_tf_data=use_tf_data,

            quick_run=quick_run,

            quick_samples=quick_samples

        )


        ########################################################
        # Save History
        ########################################################

        self.save_history(

            history,

            model_name

        )

        ########################################################
        # Evaluate
        ########################################################

        metrics = self.evaluate_model(

            model,

            history,

            X_test,

            y_test,

            class_names

        )

        print("=" * 70)

        print("Training Pipeline Completed Successfully")

        print("=" * 70)

        return model, history, metrics


############################################################
# Main
############################################################

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Fall Detection model")
    parser.add_argument("--model_name", default="cnn_lstm_improved")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_tf_data", action="store_true", default=True)
    parser.add_argument("--no-tf-data", dest="use_tf_data", action="store_false",
                        help="Disable tf.data pipeline")
    parser.add_argument("--quick_run", action="store_true", help="Use small subset for quick iteration")
    parser.add_argument("--quick_samples", type=int, default=2048)
    args = parser.parse_args()

    trainer = Trainer()

    CLASS_NAMES = None
    if os.path.exists(trainer.classes_path):
        with open(trainer.classes_path, "r") as f:
            CLASS_NAMES = json.load(f)

    trainer.run(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_names=CLASS_NAMES,
        use_tf_data=args.use_tf_data,
        quick_run=args.quick_run,
        quick_samples=args.quick_samples
    )


if __name__ == "__main__":

    main()