from sklearn.linear_model import LinearRegression
import joblib

# Training data
X = [[1], [2], [3], [4], [5]]
y = [35, 45, 55, 65, 75]

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully.")

# Make predictions
hours = [[6], [7], [8]]
predictions = model.predict(hours)

print("Predictions:", predictions)
