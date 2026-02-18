
import streamlit as st
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# DL
try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False

st.title("🚀 AutoML Tool — Upload → Train → Predict")

# ======================
# Upload Dataset
# ======================
uploaded_file = st.file_uploader("Upload Training File (Excel/CSV)", type=["xlsx","csv"])

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head())

    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")

    # ======================
    # Target selection
    # ======================
    target = st.selectbox("Select Target Column", df.columns)

    y = df[target]
    X = df.drop(columns=[target])

    # Date features
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

    # Problem type
    if y.dtype == "object" or y.nunique() < 20:
        problem_type = "classification"
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        problem_type = "regression"

    st.write("Detected Problem:", problem_type)

    # ======================
    # Model choice
    # ======================
    model_choice = st.selectbox("Choose Model Type", ["AUTO","ML","DL"])

    if st.button("Train Model"):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ======================
        # MACHINE LEARNING
        # ======================
        if model_choice == "ML" or (model_choice=="AUTO" and not DL_AVAILABLE):

            num_cols = X_train.select_dtypes(include=np.number).columns
            cat_cols = X_train.select_dtypes(exclude=np.number).columns

            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", LabelEncoder())
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_pipeline, num_cols),
                ("cat", numeric_pipeline, cat_cols)
            ])

            if problem_type == "classification":
                model = RandomForestClassifier(n_estimators=150)
            else:
                model = RandomForestRegressor(n_estimators=150)

            pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            st.success("Training Completed")

            if problem_type == "classification":
                st.write("Accuracy:", accuracy_score(y_test, preds))
                st.write("F1:", f1_score(y_test, preds, average="weighted"))
            else:
                st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
                st.write("R2:", r2_score(y_test, preds))

            joblib.dump(pipe, "automl_model.pkl")
            st.success("Model Saved")

        # ======================
        # DEEP LEARNING
        # ======================
        else:

            epochs = st.number_input("Epochs", 5, 100, 20)

            X_train_num = X_train.select_dtypes(include=np.number).fillna(0)
            X_test_num = X_test.select_dtypes(include=np.number).fillna(0)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_num)
            X_test_scaled = scaler.transform(X_test_num)

            model = keras.Sequential([
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid" if problem_type=="classification" else "linear")
            ])

            model.compile(
                optimizer="adam",
                loss="binary_crossentropy" if problem_type=="classification" else "mse",
                metrics=["accuracy"] if problem_type=="classification" else ["mae"]
            )

            es = EarlyStopping(patience=3, restore_best_weights=True)

            model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, callbacks=[es], verbose=0)

            preds = model.predict(X_test_scaled)

            st.success("DL Training Completed")

            if problem_type == "classification":
                preds = (preds > 0.5).astype(int)
                st.write("Accuracy:", accuracy_score(y_test, preds))
            else:
                st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

            model.save("automl_dl_model.h5")
            st.success("DL Model Saved")


# ======================
# PREDICTION SECTION
# ======================
st.header("🔮 Predict New File")

pred_file = st.file_uploader("Upload File for Prediction", type=["xlsx","csv"])

if pred_file and st.button("Predict"):

    model = joblib.load("automl_model.pkl")

    if pred_file.name.endswith(".xlsx"):
        new_df = pd.read_excel(pred_file)
    else:
        new_df = pd.read_csv(pred_file)

    preds = model.predict(new_df)
    new_df["Prediction"] = preds

    st.dataframe(new_df.head())

    csv = new_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
