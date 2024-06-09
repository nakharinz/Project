import streamlit as st
import pandas as pd
import joblib

st.title('Cost Prediction')

# Check if dataframe is available in session state
if 'df' in st.session_state:
    df = st.session_state['df']
    target = 'claim'

    # Check if model is available in session state
    if 'model' in st.session_state:
        model = st.session_state['model']
        st.sidebar.header('Input Features')

        # Create an empty DataFrame with the same columns as the input data
        input_data = pd.DataFrame(columns=df.drop(target, axis=1).columns)

        # Populate input features
        for col in input_data.columns:
            if col == 'age':
                # Slider for Age
                min_value = int(df[col].min())
                max_value = 100  # Update max_value to 100
                default_value = int(df[col].mean())
                input_data.loc[0, col] = st.sidebar.slider(label="Age", min_value=min_value, max_value=max_value, value=default_value)
            elif col == 'weight':
                # Slider for Weight
                min_value = int(df[col].min())
                max_value = int(df[col].max())
                default_value = int(df[col].mean())
                input_data.loc[0, col] = st.sidebar.slider(label="Weight", min_value=min_value, max_value=max_value, value=default_value)
            elif col == 'bmi':
                # Slider for BMI
                min_value = df[col].min()
                max_value = df[col].max()
                default_value = df[col].mean()
                input_data.loc[0, col] = st.sidebar.slider(label="BMI", min_value=min_value, max_value=max_value, value=default_value)
            elif col == 'hereditary_diseases':
                unique_values = df[col].unique()
                input_data.loc[0, col] = st.sidebar.selectbox(label="Genetic Disorders", options=unique_values)
            elif col == 'sex_male':
                # Radio for Sex
                selected_sex = st.sidebar.radio(label="Sex", options=['Male', 'Female'])
                # Map 'Male' to 1 and 'Female' to 0
                input_data.loc[0, col] = 1 if selected_sex == 'Male' else 0
            elif col == 'no_of_dependents':
                min_value = int(df[col].min())
                max_value = int(df[col].max())
                step = 1
                input_data.loc[0, col] = st.sidebar.number_input(label="Number of Dependents", value=int(df[col].mean()),
                                                                 min_value=min_value, max_value=max_value, step=step)
            elif col == 'smoker':
                # Radio for Sex
                selected_smoke = st.sidebar.radio(label="Smoker", options=['Yes', 'No'])
                input_data.loc[0, col] = 1 if selected_smoke == 'Yes' else 0

            elif col == 'city':
                unique_values = df[col].unique()
                input_data.loc[0, col] = st.sidebar.selectbox(label="City", options=unique_values)

            elif col == 'bloodpressure':
                # Slider for BP
                min_value = 40#int(df[col].min())
                max_value = 130#int(df[col].max())
                default_value = int(df[col].mean())
                input_data.loc[0, col] = st.sidebar.slider(label="Blood Pressure (diastolic)", min_value=min_value, max_value=max_value, value=default_value)

            elif col == 'diabetes':
                # Radio for diabetes
                selected_diabetes = st.sidebar.radio(label="Diabetic", options=['Yes', 'No'])
                input_data.loc[0, col] = 1 if selected_diabetes == 'Yes' else 0

            elif col == 'regular_ex':
                # Radio for diabetes
                selected_regular_ex = st.sidebar.radio(label="Regularly Exercise", options=['Yes', 'No'])
                input_data.loc[0, col] = 1 if selected_regular_ex == 'Yes' else 0

            elif col == 'job_title':
                unique_values = df[col].unique()
                input_data.loc[0, col] = st.sidebar.selectbox(label="Job title", options=unique_values)

            elif pd.api.types.is_numeric_dtype(df[col]):
                # Numeric inputs
                min_value = int(df[col].min())
                max_value = int(df[col].max())
                step = 1
                input_data.loc[0, col] = st.sidebar.number_input(label=col, value=int(df[col].mean()), min_value=min_value, max_value=max_value, step=step)
            else:
                # Non-numeric inputs
                unique_values = df[col].unique()
                input_data.loc[0, col] = st.sidebar.selectbox(label=col, options=unique_values)

        st.subheader('Input Data')
        st.write(input_data)

        # Make prediction
        try:
            prediction = model.predict(input_data)
            st.subheader('Prediction')
            st.write(f'Predicted insurance claim: {prediction[0]:.02f} R$')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.error('Model not found. Please train the model first.')
else:
    st.error('No data found in session state. Please upload a dataset first.')
