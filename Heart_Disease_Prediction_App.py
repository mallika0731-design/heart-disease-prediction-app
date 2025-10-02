# ---------------------------------
# Heart Disease Prediction App
# With Data Analysis, ML, Cross-Validation, and Streamlit
# ---------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import streamlit as st

# Load Data
import pandas as pd

DATA_PATH = "Heart_Disease_uci.csv"  # just the file name
df = pd.read_csv(DATA_PATH)

df = pd.read_csv(DATA_PATH)

# Rename target column if needed
if 'num' in df.columns:
    df = df.rename(columns={'num': 'target'})

# Drop columns not needed for prediction
drop_cols = ['id', 'dataset']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Data Analysis
st.title("Heart Disease Prediction App")
st.header("Data Analysis")

with st.expander("Show Data Preview"):
    st.write("First 5 rows of the dataset:")
    st.write(df.head())

    st.write("Dataset Info:")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("Summary Statistics:")
    st.write(df.describe())

# --- Additional Data Analysis ---
with st.expander("Show More Data Analysis"):
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Correlation Heatmap")
    # Only use numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('target')
    n_num = len(num_cols)
    ncols = 3
    nrows = (n_num + ncols - 1) // ncols
    fig_dist, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3*nrows))
    axes = axes.flatten() if n_num > 1 else [axes]
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    st.pyplot(fig_dist)
    plt.close(fig_dist)
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, ax=ax)
    st.pyplot(fig)
    plt.close(fig)
with st.expander("Show Target Variable Distribution"):
    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, ax=ax)
    st.pyplot(fig)

# Encode categorical features
X = df.drop('target', axis=1)
y = df['target']
X = pd.get_dummies(X, drop_first=True)

# Handle missing values by imputing with mean for numeric columns
X = X.fillna(X.mean())

# Train/Test Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning with more robust search and regularization
with st.expander("Show Hyperparameter Tuning Results"):
    st.subheader("Hyperparameter Tuning")

    # Random Forest tuning with more regularization and control
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        rf_param_grid, cv=5, n_jobs=-1, scoring='accuracy'
    )
    rf_grid.fit(X_train_scaled, y_train)
    best_rf = rf_grid.best_estimator_
    st.write("Best Random Forest Params:", rf_grid.best_params_)

    # Logistic Regression tuning with stronger regularization
    lr_param_grid = {
        'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2'],
        'class_weight': ['balanced']
    }
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        lr_param_grid, cv=5, n_jobs=-1, scoring='accuracy'
    )
    lr_grid.fit(X_train_scaled, y_train)
    best_lr = lr_grid.best_estimator_
    st.write("Best Logistic Regression Params:", lr_grid.best_params_)

# Model Training and Evaluation with cross-validation and reporting
models = {
    "Random Forest": best_rf,
    "Logistic Regression": best_lr
}

model_reports = {}
cv_scores_dict = {}

with st.expander("Show Model Comparison"):
    st.subheader("Model Comparison")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        model_reports[name] = report
        # Use stratified cross-validation for better generalization
        cv_scores = cross_val_score(
            model, scaler.transform(X), y, cv=10, scoring='accuracy'
        )
        cv_scores_dict[name] = cv_scores
        st.write(f"**{name}**")
        st.text(classification_report(y_test, y_pred))
        st.write(f"Cross-Validation Accuracy Scores: {cv_scores}")
        st.write(f"Mean CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Choose best model (Random Forest by default)
model_choice = st.selectbox("Choose Model for Prediction", list(models.keys()))
best_model = models[model_choice]
def user_input_features():
    inputs = {}
    # Use original df (not X) to get column types before encoding
    for col in df.drop('target', axis=1).columns:
        if df[col].dtype == 'object':
            options = sorted([str(x) for x in df[col].unique()])
            val = st.selectbox(f"{col}", options)
            inputs[col] = val
        elif len(df[col].unique()) <= 10:
            options = sorted([str(x) for x in df[col].unique()])
            default_idx = 0
            try:
                default_idx = options.index(str(int(df[col].mean())))
            except Exception:
                default_idx = 0
            val = st.selectbox(f"{col}", options, index=default_idx)
            inputs[col] = val
        else:
            # For continuous numeric columns, use number_input
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            val = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
            inputs[col] = val
    # Convert to DataFrame and encode to match training features
    user_df = pd.DataFrame([inputs])
    user_df_encoded = pd.get_dummies(user_df)
    # Ensure all columns from training are present, fill missing with 0
    for col in X.columns:
        if col not in user_df_encoded.columns:
            user_df_encoded[col] = 0
    # Remove any extra columns not in X
    user_df_encoded = user_df_encoded[X.columns]
    return user_df_encoded

with st.form("prediction_form"):
    user_df = user_input_features()
    submitted = st.form_submit_button("Predict")
    if submitted:
        user_scaled = scaler.transform(user_df)
        prediction = best_model.predict(user_scaled)[0]
        proba = best_model.predict_proba(user_scaled)[0][int(prediction)] if hasattr(best_model, "predict_proba") else None
        st.success(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
        if proba is not None:
            st.info(f"Prediction Probability: {proba:.2f}")

# Footer
# Custom note for editing the 'sex' column
st.markdown("""
**Note:** The 'sex' column will be handled as numeric (0 = Female, 1 = Male) and will not be one-hot encoded.
""")

# Optionally, display a mapping for user reference
if 'sex' in df.columns:
    st.write("Sex column mapping: 0 = Female, 1 = Male")

st.markdown("""
    Developed with ❤️ by [Mallika Bhardwaj]
    Email: mallika0731@gmail.com
""")
