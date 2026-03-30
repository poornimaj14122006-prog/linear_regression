import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1,1)
y = np.array([2, 4, 6, 8, 10])

# Create model
model = LinearRegression()
model.fit(X, y)

# Predict value
prediction = model.predict([[6]])

print("Prediction for 6:", prediction)

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.show()
