import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report


def get_fashion_label_dict():
    return {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot',
    }


def load_csv_rows(path):
    # Load CSV where first column is label and remaining are pixels
    df = pd.read_csv(path, header=0, low_memory=False)
    if df.shape[1] < 2:
        raise ValueError(f"CSV at {path} does not look like Fashion-MNIST format")
    y = df.iloc[:, 0].astype('int32').values
    X = df.iloc[:, 1:].astype('float32').values / 255.0
    return X, y


def build_model(input_dim=784):
    return Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])


def save_history(history, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    with open(out_dir / 'training_stats.json', 'w') as f:
        json.dump(hist, f, indent=2)
    pd.DataFrame(hist).to_csv(out_dir / 'training_stats.csv', index=False)


def parse_args():
    p = argparse.ArgumentParser(description='Train Fashion-MNIST from CSV')
    p.add_argument('--train-csv', type=str, required=True)
    p.add_argument('--test-csv', type=str, required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--out-dir', type=str, default='output')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    X_train, y_train = load_csv_rows(args.train_csv)
    X_test, y_test = load_csv_rows(args.test_csv)

    label_map = get_fashion_label_dict()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)

    model = build_model(input_dim=X_train.shape[1])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Final evaluation on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=args.batch_size, verbose=2)
    print('Final test loss:', test_loss, 'test acc:', test_acc)

    # Confusion matrix & classification report
    y_pred = np.argmax(model.predict(X_test, batch_size=args.batch_size), axis=1)
    report = classification_report(y_test, y_pred, target_names=[label_map[i] for i in range(10)])

    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    # Save model and training stats
    model.save(out_dir / 'fashion_mnist_model.keras')
    save_history(history, out_dir)

    print('Done. Model, stats, and evaluation saved to', str(out_dir))


if __name__ == '__main__':
    main()
