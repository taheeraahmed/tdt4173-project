from sklearn.metrics import mean_squared_error
import numpy as np

def training_mse(targets, predictions):
    print("MSE on training data:", mean_squared_error(targets, predictions))

def display_cross_val_scores(scores):
    mse_scores = np.sqrt(-scores)
    print("MSE scores:", mse_scores)
    print("Mean MSE:", mse_scores.mean())
    print("Std. dev:", mse_scores.std())