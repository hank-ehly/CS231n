from __future__ import print_function

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

# Load the raw CIFAR-10 data.
cifar10_dir = './cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)  # (50000, 32, 32, 3)
print('Training labels shape: ', y_train.shape)  # (50000,)
print('Test data shape: ', X_test.shape)  # (10000, 32, 32, 3)
print('Test labels shape: ', y_test.shape)  # (10000,)

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):  # 0 plane, 1 car, etc..
#     idxs = np.flatnonzero(y_train == y)  # get all y_train indices that are the current label index
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)  # get 7 of those indices
#     for i, idx in enumerate(idxs):  # 0 6407, 1 25737, etc..
#         plt_idx = i * num_classes + y + 1  # 1, 11, 21, 31 ... 2, 12, 22, 32 ...
#         plt.subplot(samples_per_class, num_classes, plt_idx)  # 7 rows, 10 columns
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))  # [0, 1, 2 ... 4999]
X_train = X_train[mask]  # first 5000 images
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))  # (5000, 32 x 32 x 3) = (5000, 3072)
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

plt.imshow(dists, interpolation='none')
plt.show()
