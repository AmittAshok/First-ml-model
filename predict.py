import joblib

# Load the trained model
model = joblib.load("model.pkl")

# New data
hours = [[6], [7], [8]]

# Make predictions
predictions = model.predict(hours)

print("Predictions:", predictions)
