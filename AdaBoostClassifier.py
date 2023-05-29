import numpy as np
from DecisionTree import DecisionTree

class AdaBoostClassifier:

    def __init__(self, n_estimators=50, learning_rate=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = []

    def fit(self, X, y):
        # Initialize the weights of the samples to be equal
        self.weights = np.ones(len(y)) / len(y)

        # Train a decision tree for each iteration
        for i in range(self.n_estimators):
            # Create a decision tree
            print ("n_estimators  ",i)
            model = DecisionTree(max_depth=2)
            model.fit(X, y)# sample_weight=self.weights)

            # Calculate the error of the decision tree
            error = 1 - np.mean(model.predict(X) == y)

            # Update the weights of the samples
            self.weights *= np.exp(-error * self.learning_rate)

            # Add the decision tree to the model
            self.models.append(model)

    def predict(self, X):
        # Make predictions for each decision tree
        predictions = np.array([model.predict(X) for model in self.models])

        # Calculate the final prediction by weighted majority vote
        prediction = np.argmax(np.dot(predictions, self.weights))

        return prediction

