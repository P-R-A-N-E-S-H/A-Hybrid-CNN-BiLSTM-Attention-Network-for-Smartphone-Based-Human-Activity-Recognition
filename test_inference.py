import os
from train import Trainer
from inference import FallDetector
import pandas as pd

if __name__ == '__main__':
    trainer = Trainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
    # take first test window
    sample = X_test[0]  # shape (200,6)
    df = pd.DataFrame(sample, columns=FallDetector.sensor_columns())
    csv_path = os.path.join(trainer.base_dir, 'data', 'test_sample.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    detector = FallDetector(
        model_path=os.path.join(trainer.model_path, 'cnn_lstm_quick_test.keras'),
        scaler_path=os.path.join(trainer.model_path, 'scaler.pkl'),
        encoder_path=os.path.join(trainer.model_path, 'label_encoder.pkl')
    )
    prediction, labels, confidence = detector.run(csv_path)
    print('Final prediction:', prediction)
    print('Labels shape:', len(labels))
