import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("train.csv")

# Drop unnecessary columns
if "ID" in df.columns:
    df.drop("ID", axis=1, inplace=True)

if "age_desc" in df.columns:
    df.drop("age_desc", axis=1, inplace=True)

# Store encoders
label_encoders = {}

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:

    if col != "Class/ASD":
        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        label_encoders[col] = le

# Target column
target = "Class/ASD"

X = df.drop(target, axis=1)
y = df[target]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Model saved successfully")
print("best_model.pkl created")
print("encoders.pkl created")