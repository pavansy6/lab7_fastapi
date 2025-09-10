from sklearn.linear_model import LinearRegression
import joblib

# Simple dataset (X = features, y = target)
X = [[1], [2], [3], [4], [5]]
y = [100, 200, 300, 400, 500]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model to file
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
