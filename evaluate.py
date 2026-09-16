"""
======================================================================
evaluate.py
======================================================================

Human Activity Recognition Using Deep Learning

UCI HAR Dataset

Models:
    1. CNN
    2. LSTM
    3. BiLSTM
    4. CNN-LSTM
    5. Proposed CNN-BiLSTM-Attention

Evaluation:
    - Test Loss
    - Test Accuracy
    - Precision
    - Recall
    - F1-Score
    - Classification Report
    - Confusion Matrix
    - Per-Class Accuracy
    - Final Model Comparison

Author:
    Pranesh
======================================================================
"""

import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# IMPORTANT:
# Import the custom layer used by the proposed model.
from models import TemporalAttentionPooling


# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =====================================================================
# DATA PATHS
# =====================================================================

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)


X_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "X_test.npy"
)


Y_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "y_test.npy"
)


# =====================================================================
# MODEL PATHS
# =====================================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "trained"
)


MODEL_PATHS = {

    "CNN":
        os.path.join(
            MODEL_DIR,
            "cnn.keras"
        ),

    "LSTM":
        os.path.join(
            MODEL_DIR,
            "lstm.keras"
        ),

    "BiLSTM":
        os.path.join(
            MODEL_DIR,
            "bilstm.keras"
        ),

    "CNN-LSTM":
        os.path.join(
            MODEL_DIR,
            "cnn_lstm.keras"
        ),

    "PROPOSED":
        os.path.join(
            MODEL_DIR,
            "proposed.keras"
        )
}


# =====================================================================
# RESULTS DIRECTORIES
# =====================================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


EVALUATION_DIR = os.path.join(
    RESULTS_DIR,
    "evaluation"
)


REPORT_DIR = os.path.join(
    EVALUATION_DIR,
    "reports"
)


CONFUSION_DIR = os.path.join(
    EVALUATION_DIR,
    "confusion_matrices"
)


os.makedirs(
    EVALUATION_DIR,
    exist_ok=True
)


os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


os.makedirs(
    CONFUSION_DIR,
    exist_ok=True
)


# =====================================================================
# CLASS NAMES
# =====================================================================

CLASS_NAMES = [

    "LAYING",

    "SITTING",

    "STANDING",

    "WALKING",

    "WALKING_DOWNSTAIRS",

    "WALKING_UPSTAIRS"

]


NUM_CLASSES = len(
    CLASS_NAMES
)


# =====================================================================
# HEADER
# =====================================================================

def print_header(
    title
):

    print()

    print(
        "=" * 80
    )

    print(
        title
    )

    print(
        "=" * 80
    )


# =====================================================================
# DEVICE INFORMATION
# =====================================================================

def show_device_information():

    print_header(
        "TENSORFLOW DEVICE INFORMATION"
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    cpus = tf.config.list_physical_devices(
        "CPU"
    )

    print(
        "CPU devices :",
        len(cpus)
    )

    print(
        "GPU devices :",
        len(gpus)
    )

    if len(gpus) == 0:

        print()

        print(
            "GPU not available."
        )

        print(
            "Evaluation will use CPU."
        )

    else:

        print()

        print(
            "GPU available."
        )

        for gpu in gpus:

            print(
                gpu
            )


# =====================================================================
# LOAD TEST DATA
# =====================================================================

def load_test_data():

    print_header(
        "LOADING TEST DATA"
    )

    # --------------------------------------------------------------
    # Check X_test
    # --------------------------------------------------------------

    if not os.path.exists(
        X_TEST_PATH
    ):

        raise FileNotFoundError(

            "\nX_test.npy not found:\n"
            + X_TEST_PATH

        )

    # --------------------------------------------------------------
    # Check y_test
    # --------------------------------------------------------------

    if not os.path.exists(
        Y_TEST_PATH
    ):

        raise FileNotFoundError(

            "\ny_test.npy not found:\n"
            + Y_TEST_PATH

        )

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    X_test = np.load(
        X_TEST_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    )

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "y_test :",
        y_test.shape
    )

    print(
        "Samples :",
        len(X_test)
    )

    print(
        "Input shape :",
        X_test.shape[1:]
    )

    return (
        X_test,
        y_test
    )


# =====================================================================
# VALIDATE TEST DATA
# =====================================================================

def validate_test_data(

    X_test,
    y_test

):

    print_header(
        "DATA VALIDATION"
    )

    passed = True

    # --------------------------------------------------------------
    # Shape
    # --------------------------------------------------------------

    if X_test.ndim == 3:

        print(
            "Shape Check : PASS"
        )

    else:

        print(
            "Shape Check : FAIL"
        )

        passed = False

    # --------------------------------------------------------------
    # Expected shape
    # --------------------------------------------------------------

    if X_test.shape[1:] == (
        128,
        9
    ):

        print(
            "Input Shape : PASS"
        )

    else:

        print(
            "Input Shape : FAIL"
        )

        passed = False

    # --------------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------------

    if len(X_test) == len(y_test):

        print(
            "Sample Count: PASS"
        )

    else:

        print(
            "Sample Count: FAIL"
        )

        passed = False

    # --------------------------------------------------------------
    # NaN
    # --------------------------------------------------------------

    if np.isnan(
        X_test
    ).any():

        print(
            "NaN Check   : FAIL"
        )

        passed = False

    else:

        print(
            "NaN Check   : PASS"
        )

    # --------------------------------------------------------------
    # Infinity
    # --------------------------------------------------------------

    if np.isinf(
        X_test
    ).any():

        print(
            "Inf Check   : FAIL"
        )

        passed = False

    else:

        print(
            "Inf Check   : PASS"
        )

    # --------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------

    unique_labels = np.unique(
        y_test
    )

    print(
        "Labels      :",
        unique_labels
    )

    if (

        unique_labels.min() >= 0

        and

        unique_labels.max()
        < NUM_CLASSES

    ):

        print(
            "Label Check : PASS"
        )

    else:

        print(
            "Label Check : FAIL"
        )

        passed = False

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    if not passed:

        raise ValueError(
            "Dataset validation failed."
        )

    print()

    print(
        "Dataset validation completed successfully."
    )


# =====================================================================
# TEST DISTRIBUTION
# =====================================================================

def print_test_distribution(
    y_test
):

    print_header(
        "TEST DATA DISTRIBUTION"
    )

    for index, name in enumerate(
        CLASS_NAMES
    ):

        count = np.sum(
            y_test == index
        )

        percentage = (

            count /
            len(y_test)

        ) * 100

        print(

            f"{name:<25}"
            f": {count:4d} "
            f"({percentage:6.2f}%)"

        )


# =====================================================================
# LOAD MODEL
# =====================================================================

def load_model(

    model_name,
    model_path

):

    print()

    print(
        f"Loading {model_name}..."
    )

    # --------------------------------------------------------------
    # Check model
    # --------------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(

            "\nModel not found:\n"
            + model_path

        )

    # --------------------------------------------------------------
    # Custom objects
    # --------------------------------------------------------------

    custom_objects = {

        "TemporalAttentionPooling":
            TemporalAttentionPooling

    }

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    model = tf.keras.models.load_model(

        model_path,

        custom_objects=custom_objects

    )

    print(
        f"{model_name} loaded successfully."
    )

    return model


# =====================================================================
# CONFUSION MATRIX
# =====================================================================

def save_confusion_matrix(

    cm,
    model_name

):

    safe_name = (

        model_name
        .lower()
        .replace(
            "-",
            "_"
        )

    )

    output_path = os.path.join(

        CONFUSION_DIR,

        f"{safe_name}_confusion_matrix.png"

    )

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    plt.figure(
        figsize=(9, 7)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        NUM_CLASSES
    )

    plt.xticks(

        tick_marks,

        CLASS_NAMES,

        rotation=45,

        ha="right"

    )

    plt.yticks(

        tick_marks,

        CLASS_NAMES

    )

    # --------------------------------------------------------------
    # Add values
    # --------------------------------------------------------------

    threshold = (
        cm.max() / 2.0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            if cm[i, j] > threshold:

                text_color = "white"

            else:

                text_color = "black"

            plt.text(

                j,

                i,

                str(
                    cm[i, j]
                ),

                horizontalalignment="center",

                color=text_color

            )

    # --------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------

    plt.ylabel(
        "True Label"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.tight_layout()

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print()

    print(
        "Confusion matrix saved:"
    )

    print(
        output_path
    )


# =====================================================================
# SAVE CLASSIFICATION REPORT
# =====================================================================

def save_report(

    model_name,
    test_loss,
    accuracy,
    precision,
    recall,
    f1,
    report,
    per_class_accuracy

):

    safe_name = (

        model_name
        .lower()
        .replace(
            "-",
            "_"
        )

    )

    output_path = os.path.join(

        REPORT_DIR,

        f"{safe_name}_report.txt"

    )

    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            "=" * 70
        )

        file.write(
            "\n"
        )

        file.write(
            f"MODEL: {model_name}\n"
        )

        file.write(
            "=" * 70
        )

        file.write(
            "\n\n"
        )

        file.write(

            f"Test Loss     : "
            f"{test_loss:.6f}\n"

        )

        file.write(

            f"Test Accuracy : "
            f"{accuracy * 100:.2f}%\n"

        )

        file.write(

            f"Precision     : "
            f"{precision * 100:.2f}%\n"

        )

        file.write(

            f"Recall        : "
            f"{recall * 100:.2f}%\n"

        )

        file.write(

            f"F1-Score      : "
            f"{f1 * 100:.2f}%\n"

        )

        file.write(
            "\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 70
        )

        file.write(
            "\n\n"
        )

        file.write(
            report
        )

        file.write(
            "\n"
        )

        file.write(
            "PER-CLASS ACCURACY\n"
        )

        file.write(
            "=" * 70
        )

        file.write(
            "\n"
        )

        for class_name, value in (

            per_class_accuracy.items()

        ):

            file.write(

                f"{class_name:<25}"
                f": {value:.2f}%\n"

            )

    print()

    print(
        "Classification report saved:"
    )

    print(
        output_path
    )


# =====================================================================
# EVALUATE ONE MODEL
# =====================================================================

def evaluate_model(

    model_name,
    model,
    X_test,
    y_test

):

    print_header(

        f"EVALUATING: {model_name}"

    )

    # --------------------------------------------------------------
    # Keras evaluation
    # --------------------------------------------------------------

    test_loss, test_accuracy = (

        model.evaluate(

            X_test,

            y_test,

            batch_size=64,

            verbose=1

        )

    )

    # --------------------------------------------------------------
    # Predictions
    # --------------------------------------------------------------

    print()

    print(
        "Generating predictions..."
    )

    probabilities = model.predict(

        X_test,

        batch_size=64,

        verbose=1

    )

    y_pred = np.argmax(

        probabilities,

        axis=1

    )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    accuracy = accuracy_score(

        y_test,

        y_pred

    )

    precision = precision_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        y_pred,

        average="weighted",

        zero_division=0

    )

    # --------------------------------------------------------------
    # Classification report
    # --------------------------------------------------------------

    report = classification_report(

        y_test,

        y_pred,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0

    )

    print()

    print(
        "Classification Report"
    )

    print(
        report
    )

    # --------------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------------

    cm = confusion_matrix(

        y_test,

        y_pred

    )

    # --------------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------------

    per_class_accuracy = {}

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        total = np.sum(
            cm[i]
        )

        correct = cm[i, i]

        if total > 0:

            class_accuracy = (

                correct /
                total

            ) * 100

        else:

            class_accuracy = 0.0

        per_class_accuracy[
            class_name
        ] = class_accuracy

    # --------------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------------

    print()

    print(
        "-" * 70
    )

    print(
        f"Model          : {model_name}"
    )

    print(

        f"Test Loss      : "
        f"{test_loss:.6f}"

    )

    print(

        f"Test Accuracy  : "
        f"{accuracy * 100:.2f}%"

    )

    print(

        f"Precision      : "
        f"{precision * 100:.2f}%"

    )

    print(

        f"Recall         : "
        f"{recall * 100:.2f}%"

    )

    print(

        f"F1-Score       : "
        f"{f1 * 100:.2f}%"

    )

    print(
        "-" * 70
    )

    # --------------------------------------------------------------
    # Per-class
    # --------------------------------------------------------------

    print()

    print(
        "Per-Class Accuracy"
    )

    for class_name, value in (

        per_class_accuracy.items()

    ):

        print(

            f"{class_name:<25}"
            f": {value:.2f}%"

        )

    # --------------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------------

    save_confusion_matrix(

        cm,

        model_name

    )

    # --------------------------------------------------------------
    # Save report
    # --------------------------------------------------------------

    save_report(

        model_name,

        test_loss,

        accuracy,

        precision,

        recall,

        f1,

        report,

        per_class_accuracy

    )

    # --------------------------------------------------------------
    # Return
    # --------------------------------------------------------------

    return {

        "model":
            model_name,

        "test_loss":
            float(
                test_loss
            ),

        "test_accuracy":
            float(
                accuracy
            ),

        "test_accuracy_percent":
            float(
                accuracy * 100
            ),

        "precision":
            float(
                precision
            ),

        "precision_percent":
            float(
                precision * 100
            ),

        "recall":
            float(
                recall
            ),

        "recall_percent":
            float(
                recall * 100
            ),

        "f1_score":
            float(
                f1
            ),

        "f1_score_percent":
            float(
                f1 * 100
            ),

        "per_class_accuracy":
            {

                key: float(value)

                for key, value
                in per_class_accuracy.items()

            },

        "confusion_matrix":
            cm.tolist(),

        "classification_report":
            report

    }


# =====================================================================
# FINAL COMPARISON
# =====================================================================

def print_final_comparison(
    results
):

    print_header(
        "FINAL MODEL COMPARISON"
    )

    print()

    print(

        f"{'MODEL':<20}"
        f"{'ACCURACY':>14}"
        f"{'PRECISION':>14}"
        f"{'RECALL':>14}"
        f"{'F1-SCORE':>14}"

    )

    print(
        "-" * 76
    )

    # --------------------------------------------------------------
    # Sort by accuracy
    # --------------------------------------------------------------

    sorted_results = sorted(

        results,

        key=lambda item:
            item[
                "test_accuracy"
            ],

        reverse=True

    )

    for result in sorted_results:

        print(

            f"{result['model']:<20}"

            f"{result['test_accuracy_percent']:>13.2f}%"

            f"{result['precision_percent']:>13.2f}%"

            f"{result['recall_percent']:>13.2f}%"

            f"{result['f1_score_percent']:>13.2f}%"

        )

    print(
        "-" * 76
    )

    # --------------------------------------------------------------
    # Best model
    # --------------------------------------------------------------

    best = sorted_results[0]

    print()

    print(
        "BEST MODEL"
    )

    print(
        "-" * 50
    )

    print(
        f"Model         : "
        f"{best['model']}"
    )

    print(

        f"Test Accuracy : "
        f"{best['test_accuracy_percent']:.2f}%"

    )

    print(

        f"Precision     : "
        f"{best['precision_percent']:.2f}%"

    )

    print(

        f"Recall        : "
        f"{best['recall_percent']:.2f}%"

    )

    print(

        f"F1-Score      : "
        f"{best['f1_score_percent']:.2f}%"

    )

    print(
        "-" * 50
    )

    return best


# =====================================================================
# SAVE JSON RESULTS
# =====================================================================

def save_results(
    results
):

    output_path = os.path.join(

        EVALUATION_DIR,

        "evaluation_results.json"

    )

    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            results,

            file,

            indent=4

        )

    print()

    print(
        "Evaluation results saved:"
    )

    print(
        output_path
    )


# =====================================================================
# TARGET CHECK
# =====================================================================

def target_analysis(
    results
):

    print_header(
        "90% ACCURACY TARGET"
    )

    # --------------------------------------------------------------
    # Proposed model
    # --------------------------------------------------------------

    proposed = None

    for result in results:

        if result["model"] == "PROPOSED":

            proposed = result

            break

    if proposed is None:

        print(
            "Proposed model result not found."
        )

        return

    accuracy = (
        proposed[
            "test_accuracy_percent"
        ]
    )

    print()

    print(
        "Proposed Model:"
    )

    print(
        "Residual CNN + SE + BiLSTM"
    )

    print(
        "+ Multi-Head Self Attention"
    )

    print(
        "+ Temporal Attention"
    )

    print()

    print(

        f"Test Accuracy : "
        f"{accuracy:.2f}%"

    )

    print(
        "Target        : 90.00%"
    )

    print()

    if accuracy >= 90.0:

        print(
            "TARGET ACHIEVED"
        )

        print(
            f"The proposed model is "
            f"{accuracy - 90.0:.2f} percentage "
            f"points above 90%."
        )

    else:

        print(
            "TARGET NOT ACHIEVED"
        )

        print(
            f"Additional improvement required: "
            f"{90.0 - accuracy:.2f} percentage points."
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()

    print(
        "=" * 80
    )

    print(
        "HUMAN ACTIVITY RECOGNITION"
    )

    print(
        "DEEP LEARNING MODEL EVALUATION"
    )

    print(
        "=" * 80
    )

    # ==============================================================
    # DEVICE
    # ==============================================================

    show_device_information()

    # ==============================================================
    # LOAD DATA
    # ==============================================================

    X_test, y_test = (
        load_test_data()
    )

    # ==============================================================
    # VALIDATE
    # ==============================================================

    validate_test_data(

        X_test,

        y_test

    )

    # ==============================================================
    # DISTRIBUTION
    # ==============================================================

    print_test_distribution(
        y_test
    )

    # ==============================================================
    # EVALUATE MODELS
    # ==============================================================

    results = []

    for model_name, model_path in (

        MODEL_PATHS.items()

    ):

        # ----------------------------------------------------------
        # Clear TensorFlow session
        # ----------------------------------------------------------

        tf.keras.backend.clear_session()

        # ----------------------------------------------------------
        # Load
        # ----------------------------------------------------------

        model = load_model(

            model_name,

            model_path

        )

        # ----------------------------------------------------------
        # Evaluate
        # ----------------------------------------------------------

        result = evaluate_model(

            model_name,

            model,

            X_test,

            y_test

        )

        # ----------------------------------------------------------
        # Store
        # ----------------------------------------------------------

        results.append(
            result
        )

        # ----------------------------------------------------------
        # Release memory
        # ----------------------------------------------------------

        del model

        tf.keras.backend.clear_session()

    # ==============================================================
    # FINAL COMPARISON
    # ==============================================================

    best = print_final_comparison(
        results
    )

    # ==============================================================
    # TARGET
    # ==============================================================

    target_analysis(
        results
    )

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================

    save_results(
        results
    )

    # ==============================================================
    # FINAL OUTPUT
    # ==============================================================

    print()

    print(
        "=" * 80
    )

    print(
        "EVALUATION COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Best Model:"
    )

    print(
        best["model"]
    )

    print()

    print(
        f"Best Test Accuracy: "
        f"{best['test_accuracy_percent']:.2f}%"
    )

    print()

    print(
        "Results folder:"
    )

    print(
        EVALUATION_DIR
    )

    print()

    print(
        "Confusion matrices:"
    )

    print(
        CONFUSION_DIR
    )

    print()

    print(
        "Classification reports:"
    )

    print(
        REPORT_DIR
    )

    print()

    print(
        "=" * 80
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()