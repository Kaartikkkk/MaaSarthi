import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ✅ Load dataset (must be named dataset.csv in your project folder)
df = pd.read_csv("dataset.csv")

# ✅ Inputs (X) and Outputs (y)
X = df[["age", "kids", "hours", "domain", "skill", "education", "city_type", "language", "device", "work_mode"]]
y_work = df["work_type"]
y_income = df["income"]

# ✅ Split columns
categorical_cols = ["domain", "skill", "education", "city_type", "language", "device", "work_mode"]
numeric_cols = ["age", "kids", "hours"]

# ✅ Preprocessor (handles NEW/UNKNOWN values safely)
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ✅ Work Recommendation Model (Classification)
work_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    ))
])

# ✅ Income Prediction Model (Regression)
income_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1
    ))
])

# ✅ Train Models
work_model.fit(X, y_work)
income_model.fit(X, y_income)

# ✅ Save models as .pkl files
joblib.dump(work_model, "work_model.pkl")
joblib.dump(income_model, "income_model.pkl")

print("✅ SUCCESS! Models saved as work_model.pkl and income_model.pkl")
