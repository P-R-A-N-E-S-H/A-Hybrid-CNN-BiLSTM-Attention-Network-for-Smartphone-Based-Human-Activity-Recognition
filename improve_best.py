"""
improve_best.py

Phase 2: improve the best model from experiments by applying class weights,
lower learning rate, and more epochs. If the best model is `cnn_lstm`, the
script will train `cnn_lstm_improved`.
"""

import os
import json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from train import Trainer


def main():
    trainer = Trainer()
    summary_path = os.path.join(trainer.result_path, "experiments_summary.json")

    if not os.path.exists(summary_path):
        print("experiments_summary.json not found. Run experiments.py first.")
        return

    with open(summary_path, "r") as f:
        results = json.load(f)

    # Select best model by accuracy
    best_model = None
    best_acc = -1.0
    for m, metrics in results.items():
        acc = metrics.get("accuracy", 0.0)
        if acc > best_acc:
            best_acc = acc
            best_model = m

    print(f"Best model from experiments: {best_model} (accuracy={best_acc:.4f})")

    # Map to improved architecture when appropriate
    target_model = best_model
    if best_model == "cnn_lstm":
        target_model = "cnn_lstm_improved"
        print("Upgrading architecture: training cnn_lstm_improved")

    # Load data to compute class weights
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()

    classes = np.unique(y_train)
    class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_array)}

    print("Computed class weights for training")

    # Improved hyperparameters
    learning_rate = 0.0005
    epochs = 100
    batch_size = 64

    model, history, metrics = trainer.run(
        model_name=target_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        class_weight=class_weights,
        use_tf_data=False,
        quick_run=True,
        quick_samples=2048
    )

    print("Improved training completed. Metrics:")
    print(metrics)

    # Save improved metrics separately
    out_path = os.path.join(trainer.result_path, f"improved_{target_model}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved improved metrics: {out_path}")


if __name__ == "__main__":
    main()
