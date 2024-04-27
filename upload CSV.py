import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Choose a CSV, XLS, or XLSX file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Please upload a file to see its content.")
else:
    file_type = uploaded_file.name.split(".")[-1]
    if file_type in ["csv"]:
      df = pd.read_csv(uploaded_file)
    elif file_type in ["xls", "xlsx"]:
      df = pd.read_excel(uploaded_file)
    else:
      st.error("Unsupported file format. Please upload a CSV, XLS, or XLSX file.")
      df = None  # Set df to None to avoid displaying empty data

    if df is not None:
        st.success("File uploaded successfully!")
        st.write(df)