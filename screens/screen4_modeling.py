"""Screen 4: Advanced Modeling Suite — ML/DL model training and predictions."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from config import get_api_key, get_provider
from agents import get_model_recommendation


def render():
    st.header("🧠 Screen 4: Advanced Modeling Suite")
    st.caption("Train ML models, get predictions, and evaluate performance.")

    api_key = get_api_key()
    provider = get_provider()

    df = st.session_state.get("clean_data") or st.session_state.get("raw_data")
    if df is None:
        st.warning("⬅️ Please upload and prepare data in earlier screens first.")
        return

    # ── Model Type Toggle ─────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        task_type = st.radio("Task Type", ["Classification", "Regression"], horizontal=True)
    with col2:
        model_family = st.radio("Model Family", ["Machine Learning", "Ensemble Methods"], horizontal=True)

    # ── Target & Features ─────────────────────────────────────────────
    st.subheader("🎯 Configure Model")

    all_cols = df.columns.tolist()
    target_col = st.selectbox("Target Column (what to predict)", all_cols)
    feature_cols = st.multiselect(
        "Feature Columns (inputs)", [c for c in all_cols if c != target_col],
        default=[c for c in all_cols if c != target_col][:10],
    )

    if not feature_cols:
        st.warning("Select at least one feature column.")
        return

    # ── Model Selection ───────────────────────────────────────────────
    if task_type == "Classification":
        if model_family == "Machine Learning":
            model_options = {"Logistic Regression": LogisticRegression, "Decision Tree": DecisionTreeClassifier}
        else:
            model_options = {"Random Forest": RandomForestClassifier, "Gradient Boosting": GradientBoostingClassifier}
    else:
        if model_family == "Machine Learning":
            model_options = {"Linear Regression": LinearRegression, "Decision Tree": DecisionTreeRegressor}
        else:
            model_options = {"Random Forest": RandomForestRegressor, "Gradient Boosting": GradientBoostingRegressor}

    model_name = st.selectbox("Select Model", list(model_options.keys()))
    test_size = st.slider("Test Split %", 10, 40, 20) / 100

    # ── AI Model Recommendation ───────────────────────────────────────
    with st.expander("🤖 AI Model Recommendation"):
        if st.button("Get AI Recommendation", disabled=not api_key):
            with st.spinner("AI is analyzing your data..."):
                rec = get_model_recommendation(
                    st.session_state.get("data_summary", ""),
                    f"{task_type} on column '{target_col}'",
                    api_key, provider,
                )
                st.markdown(rec)

    # ── Train Model ───────────────────────────────────────────────────
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner(f"Training {model_name}..."):
            try:
                # Prepare data
                model_df = df[feature_cols + [target_col]].dropna()

                # Encode categoricals
                label_encoders = {}
                for col in model_df.select_dtypes(include=["object", "category"]).columns:
                    le = LabelEncoder()
                    model_df[col] = le.fit_transform(model_df[col].astype(str))
                    label_encoders[col] = le

                X = model_df[feature_cols]
                y = model_df[target_col]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Train
                model_class = model_options[model_name]
                if model_name == "Logistic Regression":
                    model = model_class(max_iter=1000, random_state=42)
                elif model_name == "Linear Regression":
                    model = model_class()
                else:
                    model = model_class(random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Results
                st.subheader("📊 Model Results")

                if task_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.3f}")
                    m2.metric("Precision", f"{prec:.3f}")
                    m3.metric("Recall", f"{rec:.3f}")
                    m4.metric("F1 Score", f"{f1:.3f}")

                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                                    color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mse)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R² Score", f"{r2:.3f}")
                    m2.metric("MAE", f"{mae:.2f}")
                    m3.metric("RMSE", f"{rmse:.2f}")
                    m4.metric("MSE", f"{mse:.2f}")

                    # Actual vs Predicted
                    fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                                     title="Actual vs Predicted")
                    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                                  x1=y_test.max(), y1=y_test.max(),
                                  line=dict(color="red", dash="dash"))
                    st.plotly_chart(fig, use_container_width=True)

                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
                    fig = px.barh(x=importance.values, y=importance.index,
                                  labels={"x": "Importance", "y": "Feature"},
                                  title="Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

                st.session_state["model_results"] = {
                    "model": model,
                    "model_name": model_name,
                    "task_type": task_type,
                    "features": feature_cols,
                    "target": target_col,
                }
                st.success(f"✅ {model_name} trained successfully!")

            except Exception as e:
                st.error(f"Error training model: {e}")
