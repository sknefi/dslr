from typing import List

from utils import sigmoid


def train_binary_model(
    x_train: List[List[float]],
    y_binary: List[int],
    learning_rate: float,
    epochs: int,
) -> tuple[List[float], float]:
    """Train one house-versus-rest model; return its weights and bias."""
    weights: List[float] = [0.0] * len(x_train[0])
    bias                 = 0.0
    student_count        = len(x_train)

    for _ in range(epochs):
        # Collect gradients from every student before updating the weights.
        weight_gradients: List[float] = [0.0] * len(weights)
        bias_gradient = 0.0

        # Pair each student's features with their correct 0/1 label.
        for features, target in zip(x_train, y_binary):
            score = bias
            for i, value in enumerate(features):
                score += weights[i] * value
            error = sigmoid(score) - target

            # Weight gradients use error * feature; bias uses error * 1.
            for i, value in enumerate(features):
                weight_gradients[i] += error * value
            bias_gradient += error

        # Update each weight using the average gradient for this epoch.
        for i in range(len(weights)):
            weights[i] -= learning_rate * weight_gradients[i] / student_count
        bias -= learning_rate * bias_gradient / student_count

    return weights, bias
