from train import Trainer
from evaluate import ModelEvaluator, load_dl_model

if __name__ == '__main__':
    trainer = Trainer()
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
    # load quick test model
    model_path = trainer.model_path + '/cnn_lstm_quick_test.keras'
    model = load_dl_model(model_path)
    evaluator = ModelEvaluator(output_dir=trainer.result_path)
    # run evaluation non-interactively
    metrics = evaluator.evaluate_dl_model(model, X_test[:512], y_test[:512], class_names=None)
    print(metrics)
