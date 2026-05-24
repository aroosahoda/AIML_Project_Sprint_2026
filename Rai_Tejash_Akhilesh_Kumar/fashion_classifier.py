import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class FashionMNISTClassifier:
    def __init__(self):
        self.model = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    def load_and_preprocess_data(self):
        """Loads the dataset and normalizes pixel values."""
        print("Loading Fashion-MNIST dataset...")
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        
        # Normalize pixel values to be between 0 and 1
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        
        print(f"Training data shape: {self.train_images.shape}")
        print(f"Testing data shape: {self.test_images.shape}\n")

    def build_model(self):
        """Builds a simple Feedforward Neural Network."""
        print("Building the neural network...")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),      # Flatten 2D image to 1D array
            tf.keras.layers.Dense(128, activation='relu'),      # Hidden layer
            tf.keras.layers.Dense(10)                           # Output layer (10 classes)
        ])
        
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()
        print("\n")

    def train_model(self, epochs=10):
        """Trains the model on the training data."""
        print(f"Training model for {epochs} epochs...")
        self.history = self.model.fit(self.train_images, self.train_labels, epochs=epochs, validation_split=0.1)
        print("\n")

    def evaluate_model(self):
        """Evaluates the model on unseen test data."""
        print("Evaluating model on test data...")
        test_loss, test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
        print(f"\n>>> Final Test Accuracy: {test_acc*100:.2f}%\n")
        
    def analyze_misclassifications(self):
        """Generates a confusion matrix to see which items get confused."""
        print("Generating confusion matrix analysis...")
       
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(self.test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        
        
        print("\nClassification Report:")
        print(classification_report(self.test_labels, predicted_labels, target_names=self.class_names))
        
        cm = confusion_matrix(self.test_labels, predicted_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Fashion-MNIST Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
       
        plt.savefig('confusion_matrix.png')
        print(">>> Saved confusion matrix visualization as 'confusion_matrix.png'")
        plt.close()


if __name__ == "__main__":
    classifier = FashionMNISTClassifier()
    
    classifier.load_and_preprocess_data()
    classifier.build_model()
    classifier.train_model(epochs=10) # 10 epochs usually hits ~88%
    classifier.evaluate_model()
    classifier.analyze_misclassifications()