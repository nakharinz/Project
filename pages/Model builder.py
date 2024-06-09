import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
import joblib
import plotly.graph_objects as go

# Streamlit Title
st.title('Regression Model Builder')

# Check if dataframe is available in session state
if 'df' in st.session_state:
    df = st.session_state['df']
    target = 'claim'

    # Sidebar for model configuration
    st.sidebar.header('Model Configuration')

    # Dummy variable trap option
    drop_options = ['None', 'first', 'if_binary']
    drop = st.sidebar.selectbox('Dummy variable trap', drop_options)
    drop = None if drop == 'None' else drop

    # Feature and target scaling options
    scaler_options = ['None', 'StandardScaler', 'MinMaxScaler', 'QuantileTransformer']
    scaler_x = st.sidebar.selectbox('Feature scaling', scaler_options)
    scaler_y = st.sidebar.selectbox('Target scaling', scaler_options)
    scaler_x = 'passthrough' if scaler_x == 'None' else eval(scaler_x + '()')
    scaler_y = None if scaler_y == 'None' else eval(scaler_y + '()')

    # Feature selection method
    selector_options = ['None', 'f_regression', 'r_regression', 'mutual_info_regression', 'SelectFromModel']
    selector = st.sidebar.selectbox('Feature selection', selector_options)
    k = 1  # Default value for number of features

    # Display slider for number of features if applicable
    if selector in ['f_regression', 'r_regression', 'mutual_info_regression']:
        k = st.sidebar.slider('Number of features', 1, len(df.columns)-1, 1)

    if selector == 'None':
        selector = 'passthrough'
    else:
        if selector == 'SelectFromModel':
            selector = SelectFromModel(RandomForestRegressor())
        else:
            selector = SelectKBest(eval(selector), k=k)

    # Regressor selection
    regressor_options = [
        'LinearRegression', 'RandomForestRegressor', 'SVR',
         'GradientBoostingRegressor',#'DecisionTreeRegressor',
        'KNeighborsRegressor', 'ElasticNet'
    ]
    regressor = st.sidebar.selectbox('Regressor', regressor_options)
    regressor = eval(regressor + '()')

    # Preprocessing pipeline
    numeric_features = df.drop(target, axis=1).select_dtypes(include='number').columns
    categorical_features = df.drop(target, axis=1).select_dtypes(exclude='number').columns

    numeric_transformer = Pipeline(steps=[('scaler', scaler_x)])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop=drop, handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Full pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('regressor', regressor)
    ])

    # Target transformer
    target_transformer = TransformedTargetRegressor(regressor=pipe, transformer=scaler_y)

    # Train model
    try:
        with st.spinner('Training model...'):
            target_transformer.fit(df.drop(target, axis=1), df[target])
        st.success('Model trained successfully!')

        # Make predictions
        predictions = target_transformer.predict(df.drop(target, axis=1))
        st.subheader(f'Mean Absolute Error (MAE): {mean_absolute_error(df[target], predictions):.5f}')

        # Plot Actual vs Predicted values
        results_df = pd.DataFrame({'Actual': df[target], 'Predicted': predictions})
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['Predicted'], y=results_df['Actual'], mode='markers', name='Actual vs Predicted'
        ))
        fig.add_trace(go.Scatter(
            x=results_df['Predicted'], y=results_df['Predicted'], mode='lines', name='Correct Prediction',
            line=dict(color='red')
        ))

        fig.update_layout(title='Actual vs Predicted', xaxis_title='Predicted Value', yaxis_title='Actual Value')
        st.plotly_chart(fig)

        # Save model option
        st.sidebar.divider()
        if st.sidebar.button('Save model'):
            #joblib.dump(target_transformer, 'model.joblib')
            st.session_state['model'] = target_transformer
            st.sidebar.success('Model saved successfully!')


    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.error('No data found in session state. Please upload a dataset first.')
