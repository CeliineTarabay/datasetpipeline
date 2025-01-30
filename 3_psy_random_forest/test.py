import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("psych_model.pkl")

# Get feature importance
importances = model.feature_importances_
feature_names = [f"Feature {i}" for i in range(len(importances))]  # Replace with actual feature names if available

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Predicting Psychological Responses")
plt.show()
