from random import Random
from typing import List

from constants import SGD_SHUFFLE_SEED
from utils import sigmoid


def train_binary_model_sgd(
    x_train: List[List[float]],
    y_binary: List[int],
    learning_rate: float,
    epochs: int,
) -> tuple[List[float], float]:
    """Train one house-versus-rest model, updating after each student."""
    weights: List[float] = [0.0] * len(x_train[0])
    bias = 0.0
    student_indices = list(range(len(x_train)))
    random = Random(SGD_SHUFFLE_SEED)

    for _ in range(epochs):
        random.shuffle(student_indices)
        for student_index in student_indices:
            features = x_train[student_index]
            target = y_binary[student_index]
            score = bias
            for i, value in enumerate(features):
                score += weights[i] * value
            error = sigmoid(score) - target

            for i, value in enumerate(features):
                weights[i] -= learning_rate * error * value
            bias -= learning_rate * error

    return weights, bias
