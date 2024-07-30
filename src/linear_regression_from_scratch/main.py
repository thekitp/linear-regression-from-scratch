import numpy as np
import matplotlib.pyplot as plt
from linear_regression_from_scratch.functions import LinearRegression

x1 = np.arange(0, 10, 0.1)
noise = np.random.randn(*x1.shape)
y1 = 3 * x1 + 2 + noise

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
