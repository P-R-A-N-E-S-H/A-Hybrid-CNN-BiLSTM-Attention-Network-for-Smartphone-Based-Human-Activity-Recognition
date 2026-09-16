"""
run_full_train.py

Compute class weights and run a full training session for cnn_lstm_improved.
"""

import numpy as np
from sklearn.utils import class_weight
from train import Trainer


def main():
    X_train = np.load('data/processed/har/X_train.npy')
    y_train = np.load('data/processed/har/y_train.npy')

    classes = np.unique(y_train)
    cw_values = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw_values)}

    print("Class weights:", class_weights)

    trainer = Trainer()

    trainer.run(
        model_name='cnn_lstm_improved',
        epochs=100,
        batch_size=32,
        learning_rate=0.0005,
        class_weight=class_weights,
        use_tf_data=True,
        quick_run=False
    )


if __name__ == '__main__':
    main()
