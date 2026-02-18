import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# ======================
# SAFE LABEL ENCODER
# ======================
class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in X.columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str)
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            le = self.encoders.get(col)
            if le:
                X[col] = X[col].astype(str)
                X[col] = X[col].map(lambda s: '<UNK>' if s not in le.classes_ else s)
                if '<UNK>' not in le.classes_:
                    le.classes_ = np.append(le.classes_, '<UNK>')
                X[col] = le.transform(X[col])
        return X


# ======================
# 1. LOAD DATA
# ======================
file_path = input("Enter file path (Excel/CSV): ")

if file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path)
else:
    df = pd.read_csv(file_path)

df = df.drop_duplicates()
df = df.dropna(axis=1, how="all")

print("Data shape:", df.shape)


# ======================
# 2. TARGET
# ======================
print("\nColumns:")
for c in df.columns:
    print("-", c)

target_input = input("\nEnter TARGET column OR AUTO: ")

target = df.columns[-1] if target_input.upper() == "AUTO" else target_input

y = df[target]
X = df.drop(columns=[target])


# ======================
# 3. DATE FEATURES
# ======================
for col in X.columns:
    if "date" in col.lower():
        try:
            X[col] = pd.to_datetime(X[col], errors="coerce")
            X[col+"_year"] = X[col].dt.year
            X[col+"_month"] = X[col].dt.month
            X[col+"_day"] = X[col].dt.day
            X.drop(columns=[col], inplace=True)
        except:
            pass


# ======================
# 4. PROBLEM TYPE
# ======================
if y.dtype == "object" or y.nunique() < 20:
    problem_type = "classification"
    y = LabelEncoder().fit_transform(y.astype(str))
else:
    problem_type = "regression"

print("Problem:", problem_type)


# ======================
# 5. SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ======================
# 6. MODEL CHOICE
# ======================
model_choice = input("\nChoose ML / DL / AUTO: ").lower()

try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False

if model_choice == "auto":
    model_choice = "dl" if (len(df) > 50000 and DL_AVAILABLE) else "ml"

print("Using:", model_choice.upper())


# ======================
# 7. MACHINE LEARNING
# ======================
if model_choice == "ml":

    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(exclude=np.number).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", LabelEncoderWrapper())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    models = {
        "rf": RandomForestClassifier() if problem_type=="classification" else RandomForestRegressor(),
        "gb": GradientBoostingClassifier() if problem_type=="classification" else GradientBoostingRegressor(),
        "et": ExtraTreesClassifier() if problem_type=="classification" else ExtraTreesRegressor()
    }

    best_model = None
    best_score = -999

    for name, base_model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", base_model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        score = accuracy_score(y_test, preds) if problem_type=="classification" else r2_score(y_test, preds)
        print(name, "score:", score)

        if score > best_score:
            best_score = score
            best_model = pipe

    model = best_model
    preds = model.predict(X_test)

    if problem_type == "classification":
        print("\nFinal Accuracy:", accuracy_score(y_test, preds))
        print("F1:", f1_score(y_test, preds, average="weighted"))
    else:
        print("\nRMSE:", np.sqrt(mean_squared_error(y_test, preds)))
        print("R2:", r2_score(y_test, preds))


# ======================
# 8. DEEP LEARNING
# ======================
elif model_choice == "dl":

    epochs_input = input("Enter epochs OR AUTO: ")
    epochs = 20 if epochs_input.upper() == "AUTO" else int(epochs_input)

    # Use numeric only
    X_train_num = X_train.select_dtypes(include=np.number).copy()
    X_test_num = X_test.select_dtypes(include=np.number).copy()

    # Clean INF / NaN
    X_train_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_num.replace([np.inf, -np.inf], np.nan, inplace=True)

    X_train_num.fillna(X_train_num.median(), inplace=True)
    X_test_num.fillna(X_test_num.median(), inplace=True)

    y_train = pd.Series(y_train).fillna(np.median(y_train))
    y_test = pd.Series(y_test).fillna(np.median(y_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    X_train_scaled = np.nan_to_num(X_train_scaled)
    X_test_scaled = np.nan_to_num(X_test_scaled)

    model = keras.Sequential([
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid" if problem_type=="classification" else "linear")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy" if problem_type=="classification" else "mse",
        metrics=["accuracy"] if problem_type=="classification" else ["mae"]
    )

    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, callbacks=[es])

    preds = model.predict(X_test_scaled)
    preds = np.nan_to_num(preds)

    if problem_type == "classification":
        preds = (preds > 0.5).astype(int)
        print("Accuracy:", accuracy_score(y_test, preds))
    else:
        print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))


# ======================
# 9. SAVE MODEL
# ======================
save = input("\nSave model? yes/no: ")
if save.lower() == "yes":
    joblib.dump(model, "automl_pipeline.pkl")
    print("Saved → automl_pipeline.pkl")

print("\n=== PIPELINE COMPLETE ===")
