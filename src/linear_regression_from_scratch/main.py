import numpy as np
import matplotlib.pyplot as plt
from linear_regression_from_scratch.functions import LinearRegression

# Fix random seed for reproducibility
np.random.seed(42)

# Define the number of data points in each cluster
n_points_per_cluster = 50

# Define the center points of the clusters
center1 = np.array([3, 3])
center2 = np.array([5, 5])

# Generate data points for the first cluster
xx1 = np.random.rand(n_points_per_cluster) * 2 - 1  # Random values between -1 and 1
yy1 = np.random.rand(n_points_per_cluster) * 2 - 1  # Random values between -1 and 1

# Shift the data points to the center of the first cluster
xx1 += center1[0]
yy1 += center1[1]

# Generate data points for the second cluster
xx2 = np.random.rand(n_points_per_cluster) * 2 - 1  # Random values between -1 and 1
yy2 = np.random.rand(n_points_per_cluster) * 2 - 1  # Random values between -1 and 1

# Shift the data points to the center of the second cluster
xx2 += center2[0]
yy2 += center2[1]

# Combine the data points into single arrays
x1 = np.concatenate((xx1, xx2))
y1 = np.concatenate((yy1, yy2))

# x1 = np.arange(0, 10, 0.1)
# noise = np.random.randn(*x1.shape)
# y1 = 3 * x1 + 2 + noise

model1 = LinearRegression()
model1.fit(x1, y1)
y = model1.predict(x1)
print(model1.coef_)

plt.subplot(1, 2, 1)
plt.plot(x1, y1, ".b")
plt.plot(x1, y.T, ".r")
plt.title("Linear Regression with fit_intercept = True")

x2 = x1
y2 = y1

model2 = LinearRegression(fit_intercept=False)
model2.fit(x2, y2)
y = model2.predict(x2)
print(model2.coef_)

plt.subplot(1, 2, 2)
plt.plot(x2, y2, ".b")
plt.plot(x2, y.T, ".r")
plt.title("Linear Regression with fit_intercept = False")

plt.show()
