"""
experiments.py

Run experiments: train CNN, LSTM, BiLSTM, CNN-LSTM and compare metrics.

Author: Copilot
"""

import os
import json
from train import Trainer


def main():
    trainer = Trainer()

    models = ["cnn", "lstm", "bilstm", "cnn_lstm"]

    results = {}

    # Baseline hyperparameters (quick smoke test)
    epochs = 5
    batch_size = 64
    learning_rate = 0.001
    quick_run = True
    quick_samples = 2048

    for m in models:
        print("\n" + "="*70)
        print(f"Running experiment for model: {m}")
        print("="*70 + "\n")

        model, history, metrics = trainer.run(
            model_name=m,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            class_weight=None,
            use_tf_data=False,
            quick_run=quick_run,
            quick_samples=quick_samples
        )

        results[m] = metrics

    out_path = os.path.join(trainer.result_path, "experiments_summary.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\nAll experiments completed. Summary saved to:")
    print(out_path)


if __name__ == "__main__":
    main()
