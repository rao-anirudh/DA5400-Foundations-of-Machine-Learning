# Name: Anirudh Rao
# Roll No.: BE21B004


# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Reading the datasets

df = pd.read_csv('FMLA1Q1Data_train.csv', header=None)
X = np.array([df[column] for column in df.columns[:-1]])
y = np.array(df[df.columns[-1]]).reshape(len(df), 1)

test_df = pd.read_csv('FMLA1Q1Data_test.csv', header=None)
X_test = np.array([test_df[column] for column in test_df.columns[:-1]])
y_test = np.array(test_df[test_df.columns[-1]]).reshape(len(test_df), 1)

# PART 1

print("\nPART 1\n")


def least_squares(X, y):
    # Defining the analytical solution to the least squares linear regression problem

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X, X.T)), X), y)
    return w


w_ML = least_squares(X, y)
print(f"w_ML = \n{w_ML}")

# PART 2

print("\nPART 2\n")


def gradient(X, y, w):
    # Defining the gradient of the squared loss for linear regression

    return 2 * np.matmul(X, np.matmul(X.T, w)) - 2 * np.matmul(X, y)


def gradient_descent(X, y, num_iter, tol):
    # Defining the gradient descent algorithm for linear regression

    num_iter = int(num_iter)
    step_size = 1e-4

    w = np.zeros(X.shape[0]).reshape(X.shape[0], 1)

    norms = []

    for t in range(1, num_iter + 1):

        norms.append(np.linalg.norm(w - w_ML))

        grad = gradient(X, y, w)

        if np.linalg.norm(grad) < tol:
            return w, norms

        w = w - step_size * grad

    return w, norms


# Running gradient descent on the training data

w_GD, norm_history = gradient_descent(X, y, 1e5, 1e-5)
print(f"Gradient descent converged in {len(norm_history)} iterations.\n")
print(f"w_GD = \n{w_GD}")

# Plotting ||w_t - w_ML|| over t

plt.figure(dpi=150)
plt.plot(norm_history)
plt.xlabel("Iteration (t)")
plt.ylabel("||$w^t - w_{ML}$||$_2$")
plt.title("||$w^t - w_{ML}$||$_2$ during gradient descent")
plt.show()

# PART 3

print("\nPART 3\n")

# Setting a random seed for reproducible results

random.seed(5400)


def stochastic_gradient_descent(X, y, num_iter, tol, batchsize=100):
    # Defining the stochastic gradient descent algorithm for linear regression

    num_iter = int(num_iter)
    step_size = 1e-2

    w = np.zeros(X.shape[0]).reshape(X.shape[0], 1)

    weights = [w]
    norms = [np.linalg.norm(w - w_ML)]

    for t in tqdm(range(1, num_iter + 1)):

        indices = random.sample(range(X.shape[1]), batchsize)
        X_batch = X[:, indices]
        y_batch = y[indices]

        grad = gradient(X_batch, y_batch, w)

        if np.linalg.norm(grad) < tol:
            return w, norms

        w = w - step_size * grad
        weights.append(w)
        w = np.mean(weights, axis=0)
        norms.append(np.linalg.norm(w - w_ML))

    return w, norms


# Running stochastic gradient descent on the training data

w_SGD, norm_history = stochastic_gradient_descent(X, y, 1e4, 1e-5)
print(f"\nw_SGD = \n{w_SGD}\n")
print(f"||w_ML|| = {np.linalg.norm(w_ML)}")
print(f"||w_SGD|| = {np.linalg.norm(w_SGD)}")
print(f"||w_ML - w_SGD|| = {norm_history[-1]}")

# Plotting ||w_t - w_ML|| over t

plt.figure(dpi=150)
plt.plot(norm_history)
plt.xlabel("Iteration (t)")
plt.ylabel("||$w^t - w_{ML}$||$_2$")
plt.title("||$w^t - w_{ML}$||$_2$ during stochastic gradient descent")
plt.show()

# PART 4

print("\nPART 4\n")


def regularized_gradient(X, y, w, lambda_R):
    # Defining the gradient of the loss for ridge regression

    return 2 * np.matmul(X, np.matmul(X.T, w)) - 2 * np.matmul(X, y) + 2 * lambda_R * w


def regularized_gradient_descent(X, y, num_iter, tol, lambda_R=1):
    # Defining the gradient descent algorithm for ridge regression

    num_iter = int(num_iter)
    step_size = 1e-4

    w = np.zeros(X.shape[0]).reshape(X.shape[0], 1)

    for t in range(1, num_iter + 1):

        grad = regularized_gradient(X, y, w, lambda_R)

        if np.linalg.norm(grad) < tol:
            return w

        w = w - step_size * grad

    return w


random.seed(5400)


def make_folds(X, y, num_folds):
    # Making k-folds of the training data

    folds = {k: None for k in range(num_folds)}
    fold_size = X.shape[1] // num_folds

    for k in range(num_folds - 1):
        indices = random.sample(range(X.shape[1]), fold_size)
        X_fold = X[:, indices]
        y_fold = y[indices]
        folds[k] = (X_fold, y_fold)
        X = np.delete(X, indices, axis=1)
        y = np.delete(y, indices, axis=0)

    folds[num_folds - 1] = (X, y)

    return folds


def loss(y_true, y_pred):
    # Defining the loss as the mean squared error

    return (1 / y_true.shape[0]) * np.linalg.norm(y_true - y_pred) ** 2


def cross_validate(X, y, num_folds, lambdas=[0]):
    # Defining the k-fold cross-validation procedure for ridge regression

    validation_losses = {lambda_R: None for lambda_R in lambdas}

    folds = make_folds(X, y, num_folds)

    for lambda_R in tqdm(lambdas):

        losses = []

        for k in range(num_folds):
            train_folds = [folds[i] for i in range(num_folds) if i != k]
            X_train = np.concatenate([train_folds[i][0] for i in range(num_folds - 1)], axis=1)
            y_train = np.concatenate([train_folds[i][1] for i in range(num_folds - 1)], axis=0)

            X_val = folds[k][0]
            y_val = folds[k][1]

            w = regularized_gradient_descent(X_train, y_train, 1e5, 1e-10, lambda_R)

            losses.append(loss(y_val, np.matmul(X_val.T, w)))

        validation_losses[lambda_R] = np.mean(losses)

    return validation_losses


# Running 5-fold cross-validation to find the best regularization parameter

lambdas = [10 ** i for i in range(-5, 4)]
cv_losses = cross_validate(X, y, 5, lambdas)
plt.figure(dpi=150)
plt.plot(lambdas, list(cv_losses.values()))
plt.xscale("log")
plt.xlabel("Regularization parameter λ")
plt.ylabel("Cross-validation loss")
plt.title("Cross-validation loss vs. regularization parameter")
plt.show()

lambdas = np.linspace(1, 1000, 1000)
cv_losses = cross_validate(X, y, 5, lambdas)
plt.figure(dpi=150)
plt.plot(lambdas, list(cv_losses.values()))
plt.xlabel("Regularization parameter λ")
plt.ylabel("Cross-validation loss")
plt.title("Cross-validation loss vs. regularization parameter")
plt.show()

best_lambda = min(cv_losses, key=cv_losses.get)
print(f"\nBest λ = {best_lambda}")

# Comparing the analytical and ridge losses on the test data

w_ridge = regularized_gradient_descent(X, y, 1e5, 1e-10, best_lambda)
print(f"\nw_R = \n{w_ridge}\n")
print(f"||w_ML|| = {np.linalg.norm(w_ML)}")
print(f"||w_R|| = {np.linalg.norm(w_ridge)}")
print(f"MSE(w_ML) = {loss(y_test, np.matmul(X_test.T, w_ML))}")
print(f"MSE(w_R) = {loss(y_test, np.matmul(X_test.T, w_ridge))}")

# PART 5

print("\nPART 5\n")

# Plotting the distribution of the data

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[1], X[0], y, c=y)
ax.set_xlabel('X2')
ax.set_ylabel('X1')
ax.set_zlabel('y')
plt.title('Distribution of the training data')
fig.colorbar(scatter, ax=ax, label='y')
plt.show()


def gaussian_kernel(x1, x2, sigma):
    # Defining the Gaussian kernel

    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))


def make_gram_matrix(X, sigma):
    # Defining the Gram matrix

    gram_matrix = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            gram_matrix[i, j] = gaussian_kernel(X[:, i], X[:, j], sigma)
    return gram_matrix


def predict(X, alpha, test, sigma):
    # Defining the kernel regression algorithm

    y = 0

    for i in range(X.shape[1]):
        y += alpha[i] * gaussian_kernel(X[:, i], test, sigma)

    return y


def kernel_cross_validate(X, y, num_folds, sigmas=[1]):
    # Defining the 5-fold cross-validation procedure for kernel regression

    validation_losses = {sigma: None for sigma in sigmas}

    folds = make_folds(X, y, num_folds)

    for sigma in tqdm(sigmas):

        losses = []

        for k in range(num_folds):
            train_folds = [folds[i] for i in range(num_folds) if i != k]
            X_train = np.concatenate([train_folds[i][0] for i in range(num_folds - 1)], axis=1)
            y_train = np.concatenate([train_folds[i][1] for i in range(num_folds - 1)], axis=0)

            X_val = folds[k][0]
            y_val = folds[k][1]

            K = make_gram_matrix(X_train, sigma)
            alpha = np.matmul(np.linalg.inv(K), y_train)
            y_val_pred = np.array([predict(X_train, alpha, X_val[:, i], sigma) for i in range(X_val.shape[1])])
            y_val_pred = y_val_pred.reshape(len(y_val_pred), 1)

            losses.append(loss(y_val, y_val_pred))

        validation_losses[sigma] = np.mean(losses)

    return validation_losses


# Performing 5-fold cross-validation on the training data to find the best kernel parameter

sigmas = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
cv_losses = kernel_cross_validate(X, y, 5, sigmas)
plt.figure(dpi=150)
plt.plot(sigmas, list(cv_losses.values()))
plt.xlabel("Gaussian kernel parameter σ")
plt.ylabel("Cross-validation loss")
plt.title("Cross-validation loss vs. Gaussian kernel parameter")
plt.show()

sigma = min(cv_losses, key=cv_losses.get)
print(f"\nBest Gaussian kernel σ = {sigma}")

# Computing the loss of kernel regression on the test data

K = make_gram_matrix(X, sigma)
alpha = np.matmul(np.linalg.inv(K), y)
y_test_pred = np.array([predict(X, alpha, X_test[:, i], sigma) for i in range(X_test.shape[1])])
y_test_pred = y_test_pred.reshape(len(y_test_pred), 1)
print(f"Kernel regression loss = {loss(y_test, y_test_pred)}")
