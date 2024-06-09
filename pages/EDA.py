import streamlit as st
import pandas as pd
import plotly.express as px

# Set page title and favicon
st.set_page_config(page_title="Exploratory Data Analysis", page_icon=":bar_chart:")

# Title
st.title('Exploratory Data Analysis')

# Load data or use sample data
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv("_insurance_data.csv")  # Replace with your data

df = st.session_state.df

# Sidebar for selecting columns and displaying details
st.sidebar.subheader("Columns Correlation")

# Select columns for correlations
selected_col = st.sidebar.multiselect('Select columns to illustrate correlations', df.columns.tolist(), default=df.columns.tolist())


st.subheader(f"DATA")
if len(selected_col) == 0:
    selected_col = df.columns
st.write(df[selected_col])

# Correlation
st.sidebar.subheader("Column Details")
corr = df[selected_col].corr(numeric_only=True).round(2)
fig_corr = px.imshow(corr, text_auto=True)
st.subheader(f"Correlations")
st.plotly_chart(fig_corr)

# Display details of selected column
col = st.sidebar.selectbox('Select a column to display details', df.columns.tolist())

st.subheader(f"Details of column: {col}")
tmp = df[col]

# Check if the column is numeric
if pd.api.types.is_numeric_dtype(tmp):
    # Outlier option
    outliers = st.sidebar.checkbox('Remove Outliers', False)

    if outliers:
        q_low = tmp.quantile(0.01)
        q_high = tmp.quantile(0.99)
        tmp = tmp[(tmp > q_low) & (tmp < q_high)]

    # Display statistics and histogram
    st.write(tmp.describe())

    # Calculate mean
    mean_value = tmp.mean()

    # Histogram plot
    fig_hist = px.histogram(tmp, x=col, marginal="box", nbins=20, title=f"Distribution")

    # Add mean line
    fig_hist.add_vline(x=mean_value, line=dict(color="red", width=1.5, dash="dash"))

    # Add mean value annotation
    fig_hist.update_layout(annotations=[
        dict(
            x=mean_value,
            y=0.95,
            xref="x",
            yref="paper",
            text=f"Mean: {mean_value:.2f}",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
            font=dict(color="red")
        )
    ])

    st.plotly_chart(fig_hist)

else:
    # For non-numeric columns, display value counts and pie chart
    st.write(tmp.value_counts())
    fig_pie = px.pie(tmp, names=col, title=f"Distribution")
    st.plotly_chart(fig_pie)
