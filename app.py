
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Logistic Regression",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Logistic Regression Classification")
st.write("Interactive Machine Learning Classification Dashboard")

st.divider()

# DATASET UPLOAD
st.sidebar.header("Dataset Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
else:
    try:
        dataset = pd.read_csv(
            r"C:\Users\ybali\Downloads\logit classification.csv"
        )
        st.sidebar.success("Default dataset loaded")
    except FileNotFoundError:
        st.info("Please upload a CSV dataset.")
        st.stop()

# DATASET OVERVIEW
st.subheader("Dataset Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Rows", dataset.shape[0])
c2.metric("Total Columns", dataset.shape[1])
c3.metric(
    "Missing Values",
    int(dataset.isnull().sum().sum())
)

with st.expander("View Dataset"):
    st.dataframe(dataset, use_container_width=True)

# FEATURE SELECTION
st.sidebar.header("Model Settings")

numeric_columns = dataset.select_dtypes(
    include=np.number
).columns.tolist()

target = st.sidebar.selectbox(
    "Select Target Column",
    dataset.columns,
    index=len(dataset.columns)-1
)

features = st.sidebar.multiselect(
    "Select Input Features",
    [c for c in numeric_columns if c != target],
    default=[
        c for c in numeric_columns if c != target
    ][:2]
)

test_size = st.sidebar.slider(
    "Test Size (%)", 10, 40, 20
)

c_value = st.sidebar.number_input(
    "Regularization C",
    min_value=0.01,
    value=1.0
)

if not features:
    st.warning("Select at least one input feature.")
    st.stop()

# DATA PREPROCESSING
data = dataset[features + [target]].dropna()

X = data[features]
y = data[target]

if y.nunique() != 2:
    st.error("This application requires a binary target.")
    st.stop()

if y.value_counts().min() < 2:
    st.error("Each class must contain at least two rows.")
    st.stop()

if st.button("🚀 Train Logistic Regression Model"):

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=0,
            stratify=y
        )

        # FEATURE SCALING
        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # MODEL TRAINING
        classifier = LogisticRegression(
            penalty="l2",
            C=c_value,
            solver="liblinear"
        )

        classifier.fit(X_train, y_train)

        # PREDICTION
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)[:, 1]

        # EVALUATION
        accuracy = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(
            y_test, y_pred,
            labels=classifier.classes_
        )

        auc = roc_auc_score(
            y_test, y_prob
        )

        fpr, tpr, thresholds = roc_curve(
            y_test, y_prob,
            pos_label=classifier.classes_[1]
        )

        # SAVE RESULTS IN SESSION
        st.session_state["trained"] = True
        st.session_state["model"] = classifier
        st.session_state["scaler"] = sc
        st.session_state["features"] = features
        st.session_state["accuracy"] = accuracy
        st.session_state["auc"] = auc
        st.session_state["cm"] = cm
        st.session_state["classes"] = classifier.classes_
        st.session_state["report"] = classification_report(
            y_test, y_pred, zero_division=0
        )
        st.session_state["fpr"] = fpr
        st.session_state["tpr"] = tpr

        st.success("Model trained successfully!")

    except ValueError as e:
        st.error(f"Training failed: {e}")

# DISPLAY RESULTS
if st.session_state.get("trained", False):

    st.divider()
    st.header("Model Performance")

    c1, c2 = st.columns(2)

    c1.metric(
        "Accuracy",
        f"{st.session_state['accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "AUC Score",
        f"{st.session_state['auc']:.3f}"
    )

    # CONFUSION MATRIX
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    display = ConfusionMatrixDisplay(
        confusion_matrix=st.session_state["cm"],
        display_labels=st.session_state["classes"]
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)
    plt.close(fig)

    # CLASSIFICATION REPORT
    st.subheader("Classification Report")

    st.code(st.session_state["report"])

    # ROC CURVE
    st.subheader("ROC Curve")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        st.session_state["fpr"],
        st.session_state["tpr"],
        label=f"AUC = {st.session_state['auc']:.2f}"
    )

    ax.plot([0, 1], [0, 1], "k--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid()

    st.pyplot(fig)
    plt.close(fig)

    # MODEL DOWNLOAD
    import joblib

    model_data = {
        "model": st.session_state["model"],
        "scaler": st.session_state["scaler"],
        "features": st.session_state["features"]
    }

    import io

    buffer = io.BytesIO()
    joblib.dump(model_data, buffer)

    st.download_button(
        "Download Trained Model",
        data=buffer.getvalue(),
        file_name="logistic_model.pkl"
    )

st.divider()
st.caption("Built with Python, Scikit-learn and Streamlit")



plt.grid()
plt.show()
