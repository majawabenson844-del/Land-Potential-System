import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
path = "augmented_data.csv"
df = pd.read_csv(path)

# 2. Rename duplicate columns
df.columns = ['Decision', 'Soil.Texture', 'Soil.Colour', 'Geological.Features', 'Elevation',
              'Natural.vegitation..tree..vigour', 'Natural.vegitation..tree..height', 'Drainage.Density']

# 3. Encode target
df["Decision"] = df["Decision"].str.strip()
y = df["Decision"].map({"High Potential": 1, "Low Potential": 0})

# 4. Encode predictors using OrdinalEncoder
X = df.drop("Decision", axis=1)
encoder = OrdinalEncoder()
X_encoded = pd.DataFrame(encoder.fit_transform(X), columns=X.columns)

# 5. Boruta feature selection
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
boruta_selector = BorutaPy(rf, n_estimators="auto", random_state=42)
boruta_selector.fit(X_encoded.values, y.values)

important_features = X_encoded.columns[boruta_selector.support_].tolist()
print("Important predictors (Čvariable-level):", important_features)

X_selected = X_encoded[important_features]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train SVM
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train_scaled, y_train)

# 9. Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 10. Save model + scaler + encoder
joblib.dump(model, "svm_model_boruta.pkl")
joblib.dump(scaler, "scaler_boruta.pkl")
joblib.dump(encoder, "ordinal_encoder.pkl")
