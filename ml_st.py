import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Function to train and evaluate models
def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2, predictions

# Streamlit App
st.title("Regression Model Comparison")
st.write("Upload your dataset, handle null values, select a target variable, and compare regression models!")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df)

    # Handle missing values
    st.write("Handling Missing Values:")
    null_handling_method = st.radio(
        "Choose a method to handle null values:",
        options=["Drop Rows with Null Values", "Fill Null Values with Median"],
        index=0,
    )

    if null_handling_method == "Drop Rows with Null Values":
        df = df.dropna()
        st.write("Rows with null values removed.")
    elif null_handling_method == "Fill Null Values with Median":
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        st.write("Null values filled with median for numeric columns and mode for categorical columns.")

    st.write("Dataset After Handling Null Values:")
    st.dataframe(df)

    # Handle Categorical Variables
    st.write("Handle Categorical Features:")
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if len(categorical_cols) > 0:
        st.write(f"Categorical Columns Detected: {categorical_cols}")
        encoding_method = st.radio(
            "Choose encoding method:",
            options=["Label Encoding", "One-Hot Encoding"],
            index=0,
        )

        # Apply encoding
        if encoding_method == "Label Encoding":
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
            st.write("Applied Label Encoding.")
        elif encoding_method == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            st.write("Applied One-Hot Encoding.")

    else:
        st.write("No categorical columns detected.")

    # Select target variable
    target = st.selectbox("Select the target variable", df.columns)

    # Select features
    features = st.multiselect("Select feature columns", df.columns, default=[col for col in df.columns if col != target])

    if target and features:
        # Split dataset
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select models
        st.sidebar.write("Choose Models to Compare")
        linear = st.sidebar.checkbox("Linear Regression", value=True)
        random_forest = st.sidebar.checkbox("Random Forest")
        decision_tree = st.sidebar.checkbox("Decision Tree")

        # Train and evaluate selected models
        results = {}
        if linear:
            mse, r2, _ = train_model(LinearRegression(), X_train, X_test, y_train, y_test)
            results["Linear Regression"] = {"MSE": mse, "R2": r2}
        if random_forest:
            mse, r2, _ = train_model(RandomForestRegressor(n_estimators=100, random_state=42), X_train, X_test, y_train, y_test)
            results["Random Forest"] = {"MSE": mse, "R2": r2}
        if decision_tree:
            mse, r2, _ = train_model(DecisionTreeRegressor(random_state=42), X_train, X_test, y_train, y_test)
            results["Decision Tree"] = {"MSE": mse, "R2": r2}

        # Display results
        st.write("Model Performance:")
        st.dataframe(pd.DataFrame(results).T)

        # Optionally, show predictions
        st.write("Sample Predictions (First 5 rows):")
        _, _, predictions = train_model(LinearRegression(), X_train, X_test, y_train, y_test)
        sample_results = pd.DataFrame({"Actual": y_test.values[:5], "Predicted": predictions[:5]})
        st.dataframe(sample_results)
