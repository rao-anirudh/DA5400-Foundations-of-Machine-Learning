# Name: Anirudh Rao
# Roll No: BE21B004

# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
from scipy.spatial import Voronoi, voronoi_plot_2d

# Loading MNIST

# Note: This may require the user to install some additional dependencies - huggingface_hub, fsspec, pyarrow

print('\nPart 1\n\nLoading MNIST\n')
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
mnist = pd.read_parquet('hf://datasets/ylecun/mnist/' + splits['train'])

# Sampling 100 images of each digit

print('Sampling images\n')
mnist_sample = pd.DataFrame()
for label in mnist['label'].unique():
    mnist_sample = pd.concat([mnist_sample, mnist[mnist['label'] == label].sample(100, random_state=5400)])
mnist_sample.index = range(len(mnist_sample))

# Reading the images

mnist_sample['image'] = mnist_sample['image'].apply(lambda x: Image.open(io.BytesIO(x['bytes'])))
mnist_sample['data'] = mnist_sample['image'].apply(lambda x: np.array(x).reshape(1, 28 * 28))
mnist_data = np.array(mnist_sample['data'].to_list()).reshape(len(mnist_sample), 28 * 28)

# Mean centering

print('Mean centering\n')
mean_centered_mnist = mnist_data - mnist_data.mean(axis=0)
mean_centered_mnist = mean_centered_mnist.T

# Visualising the digits

fig, ax = plt.subplots(2, 5, dpi=150, figsize=(10, 5))
labels = sorted(mnist_sample['label'].unique())
indices = []

for i in range(len(labels)):
    image_index = mnist_sample[mnist_sample['label'] == labels[i]].sample(1, random_state=5400).index[0]
    indices.append(image_index)
    ax[i // 5, i % 5].imshow(mean_centered_mnist[:, image_index].reshape(28, 28), cmap='gray')
    ax[i // 5, i % 5].axis('off')

plt.tight_layout()
plt.show()

# Computing the principal components

print('Performing principal component analysis\n')
covariance_matrix = (1 / len(mnist_sample)) * np.matmul(mean_centered_mnist, mean_centered_mnist.T)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
eigenvalues = np.flip(eigenvalues)
eigenvectors = np.flip(eigenvectors, axis=1)
eigenvectors = eigenvectors.T

# Visualising the principal components

fig, ax = plt.subplots(2, 5, dpi=150, figsize=(10, 5))
pcs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in range(len(pcs)):
    image = eigenvectors[pcs[i] - 1, :].reshape(28, 28)
    ax[i // 5, i % 5].imshow(image, cmap='gray')
    ax[i // 5, i % 5].axis('off')
    ax[i // 5, i % 5].set_title(f'PC {pcs[i]}')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 5, dpi=150, figsize=(10, 5))
pcs = [20, 50, 100, 200, 250, 300, 400, 500, 600, 784]

for i in range(len(pcs)):
    image = eigenvectors[pcs[i] - 1, :].reshape(28, 28)
    ax[i // 5, i % 5].imshow(image, cmap='gray')
    ax[i // 5, i % 5].axis('off')
    ax[i // 5, i % 5].set_title(f'PC {pcs[i]}')

plt.tight_layout()
plt.show()

# Computing the explained variance

explained_variance_ratio = 100 * eigenvalues / np.sum(eigenvalues)
print(f'PC 1 explains {explained_variance_ratio[0]:.1f}% of the variance.')
print(f'PC 2 explains {explained_variance_ratio[1]:.1f}% of the variance.')
print(f'PC 3 explains {explained_variance_ratio[2]:.1f}% of the variance.\n')

plt.figure(dpi=150)
plt.plot(explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.show()


# Reconstructing digits

def reconstruct_digit(x, eigenvectors, d):
    scalars = [np.matmul(x, eigenvector) for eigenvector in eigenvectors[:d]]
    reconstructed_digit = np.sum([scalar * eigenvector for scalar, eigenvector in zip(scalars, eigenvectors[:d])],
                                 axis=0)
    return reconstructed_digit.reshape(28, 28)


fig, ax = plt.subplots(10, 6, dpi=150, figsize=(10, 20))
pcs = [1, 10, 100, 500, 784]

for i in range(len(indices)):
    x = mean_centered_mnist[:, indices[i]]
    for j in range(len(pcs) + 1):
        if j == 0:
            ax[i, j].imshow(x.reshape(28, 28), cmap='gray')
            ax[i, j].axis('off')
            ax[i, j].set_title('Original')
        else:
            reconstructed_digit = reconstruct_digit(x, eigenvectors, pcs[j - 1])
            ax[i, j].imshow(reconstructed_digit, cmap='gray')
            ax[i, j].axis('off')
            ax[i, j].set_title(f'PC {pcs[j - 1]}')

plt.tight_layout()
plt.show()

# Computing the cumulative explained variance and plotting at the 90% threshold

cumulative_variance = np.cumsum(explained_variance_ratio)
threshold = np.argmax(cumulative_variance >= 90)
print(f'The first {threshold} PCs explain 90% of the variance.\n')

fig, ax = plt.subplots(2, 5, dpi=150, figsize=(10, 5))

for i in range(len(indices)):
    image = reconstruct_digit(mean_centered_mnist[:, indices[i]], eigenvectors, threshold)
    ax[i // 5, i % 5].imshow(image, cmap='gray')
    ax[i // 5, i % 5].axis('off')

plt.tight_layout()
plt.show()

# Loading the data for k-means

print('\nPart 2\n\nLoading dataset\n')
df = pd.read_csv('cm_dataset_2.csv', header=None)
X = df.to_numpy()

plt.figure(dpi=150)
plt.scatter(df[0], df[1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# Lloyd's algorithm

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def assign_cluster(x, centroids):
    distances = np.array([distance(x, centroid) for centroid in centroids])
    return np.argmin(distances)


def compute_centroids(X, assignments):
    clusters = sorted(np.unique(assignments))

    return np.array([X[assignments == i].mean(axis=0) for i in clusters])


def error(X, assignments, centroids):
    return np.sum([distance(x, centroids[assignment]) for x, assignment in zip(X, assignments)])


def lloyd_algorithm(X, k=2, initial_centroids=None):
    np.random.seed(5400)

    if initial_centroids is None:
        centroids = X[np.random.choice(len(X), k, replace=False)]
    else:
        centroids = initial_centroids

    assignments = np.array([assign_cluster(x, centroids) for x in X])
    errors = [error(X, assignments, centroids)]

    while True:
        new_centroids = compute_centroids(X, assignments)
        new_assignments = np.array([assign_cluster(x, new_centroids) for x in X])
        errors.append(error(X, new_assignments, new_centroids))

        if np.all(new_assignments == assignments):
            assignments = new_assignments
            break

        assignments = new_assignments

    return assignments, errors


def plot_clusters(X, assignments):
    plt.figure(dpi=150)
    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


# Running Lloyd's for different random initializations with k = 2

print('Running Lloyd for k = 2 with 5 different random initializations\n')

np.random.seed(5400)

initializations = [X[np.random.choice(len(X), 2, replace=False)] for i in range(5)]
cluster_data = []

plt.figure(dpi=150)

for i in range(1, 6):
    clusters, errors = lloyd_algorithm(X, 2, initializations[i - 1])
    cluster_data.append(clusters)
    plt.plot(errors, label=f'Initialization {i}')

plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()

fig, ax = plt.subplots(5, 1, dpi=150, figsize=(5, 15))

for i in range(len(cluster_data)):
    ax[i].scatter(X[:, 0], X[:, 1], c=cluster_data[i])
    ax[i].set_title(f'Initialization {i + 1}')
    ax[i].set_xlabel('x1')
    ax[i].set_ylabel('x2')

plt.tight_layout()
plt.show()


# Plotting Voronoi regions for k = 2, 3, 4, 5

def plot_voronoi(X, assignments):
    centroids = compute_centroids(X, assignments)

    plt.figure(dpi=150)
    plt.title(f'$k$ = {len(centroids)}')
    plt.scatter(X[:, 0], X[:, 1], c=assignments, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.xlabel('x1')
    plt.ylabel('x2')

    if len(centroids) > 2:

        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, plt.gca(), show_vertices=False, show_points=False)
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    elif len(centroids) == 2:

        midpoint = (centroids[0] + centroids[1]) / 2
        dx, dy = centroids[1] - centroids[0]

        if dx != 0:
            slope = dy / dx
            perp_slope = -1 / slope
            x_vals = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
            y_vals = perp_slope * (x_vals - midpoint[0]) + midpoint[1]
        else:
            x_vals = np.array([midpoint[0], midpoint[0]])
            y_vals = np.array([X[:, 1].min() - 1, X[:, 1].max() + 1])

        plt.plot(x_vals, y_vals, c='black', ls='--', lw=0.9)
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    plt.show()


print('Plotting Voronoi regions for k = 2, 3, 4, 5\n')

for k in [2, 3, 4, 5]:
    clusters, errors = lloyd_algorithm(X, k)
    plot_voronoi(X, clusters)
