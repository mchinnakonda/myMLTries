"""
==============================================
Face completion with a multi-output estimators
==============================================

This example shows the use of multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

The first column of images shows true faces. The next columns illustrate
how extremely randomized trees, k nearest neighbors, linear
regression and ridge regression complete the lower half of those faces.

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

# Load the faces datasets
data = fetch_olivetti_faces()
targets = data.target

data = data.images.reshape((len(data.images), -1))
train = data[targets < 30]
test = data[targets >= 30]  # Test on independent people

# Test on a subset of people
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

n_pixels = data.shape[1]
X_train = train[:, :np.ceil(0.5 * n_pixels)]  # Upper half of the faces
y_train = train[:, np.floor(0.5 * n_pixels):]  # Lower half of the faces
X_test = test[:, :np.ceil(0.5 * n_pixels)]
y_test = test[:, np.floor(0.5 * n_pixels):]


# list of integers to split the images vertically
leftHalfIds = []
rightHalfIds = []

for i in range(64):
    for j in range(32):
        leftHalfIds.append(j + i*64)
        rightHalfIds.append(j+32 + i*64)
leftHalfIds = tuple(leftHalfIds)
rightHalfIds = tuple(rightHalfIds)

# Split training and test data vertically
X_train2 = []
y_train2 = []
X_test2 = []
y_test2 = []

for i in range(300):
    tempListX = []
    tempListY = []    
    for j in leftHalfIds:
        tempListX.append(train[i,j])
    X_train2.append(tempListX)
    for k in rightHalfIds:
        tempListY.append(train[i,k])
    y_train2.append(tempListY)
    
for i in range(5):
    tempListX = []
    tempListY = []    
    for j in leftHalfIds:
        tempListX.append(test[i,j])
    X_test2.append(tempListX)
    for k in rightHalfIds:
        tempListY.append(test[i,k])
    y_test2.append(tempListY)

X_train2 = np.array(X_train2)
y_train2 = np.array(y_train2)
X_test2 = np.array(X_test2)
y_test2 = np.array(y_test2)



# Fit estimators
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train2, y_train2)
    y_test_predict[name] = estimator.predict(X_test2)

# Plot the completed faces
image_shape = (64, 64)   
    

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = []
    for p in range(64):
        true_face.append(X_test2[i][p*32: p*32 + 32])
        true_face.append(y_test2[i][p*32: p*32 + 32])
    
    true_face = np.concatenate(true_face)
    true_face_old = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")


    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = []
        for p in range(64):
            completed_face.append(X_test2[i][p*32: p*32 + 32])
            completed_face.append(y_test_predict[est][i][p*32: p*32 + 32])
    
        completed_face = np.concatenate(completed_face)


        #completed_face = np.hstack((X_test2[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()
