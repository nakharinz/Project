import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder
from sklearn.feature_selection import f_regression, r_regression, mutual_info_regression, SelectKBest, SelectFromModel
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error

# Function to calculate MAPE
def mean_absolute_percentage_error(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Streamlit Title
st.title('Compare MAE from different regressors')
st.subheader(f"(One-hot encoder & Standard scaler)")
# Check if dataframe is available in session state
if 'df' in st.session_state:
    df = st.session_state['df']
    target = 'claim'

    # Predefined regressors
    regressors = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(),
        'SVR': SVR(),
        #'DecisionTreeRegressor': DecisionTreeRegressor(),
        'GradientBoostingRegressor': GradientBoostingRegressor(),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'ElasticNet': ElasticNet()
    }

    # Table to display algorithm and MAE
    results = []

    # Train each regressor and calculate MAE
    for regressor_name, regressor in regressors.items():
        try:
            # Preprocessing pipeline
            numeric_features = df.drop(target, axis=1).select_dtypes(include='number').columns
            categorical_features = df.drop(target, axis=1).select_dtypes(exclude='number').columns

            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

            # Full pipeline
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', regressor)
            ])

            # Target transformer
            target_transformer = TransformedTargetRegressor(regressor=pipe)

            # Train model
            target_transformer.fit(df.drop(target, axis=1), df[target])

            # Make predictions
            predictions = target_transformer.predict(df.drop(target, axis=1))
            mae = mean_absolute_error(df[target], predictions)
            # Calculate MAPE using the custom function
            mape = mean_absolute_percentage_error(df[target], predictions)

            # Append results
            results.append({'Algorithm': regressor_name, 'MAE': mae, 'MAPE': mape})

        except Exception as e:
            st.error(f"An error occurred with {regressor_name}: {e}")

    # Display results as a table
    st.subheader('Regression Algorithms Performance')
    df_results = pd.DataFrame(results)
    st.table(df_results)
    st.write("mae : mean absolute error")
    st.write("mape : mean absolute percentage error")
else:
    st.error('No data found in session state. Please upload a dataset first.')
