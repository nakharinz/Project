import streamlit as st
from sklearn.datasets import load_diabetes, load_iris
import altair as alt
from sklearn.feature_selection import (GenericUnivariateSelect, r_regression, f_regression, mutual_info_regression,
                                       chi2, f_classif, mutual_info_classif)
import pandas as pd

mode = st.sidebar.selectbox("Mode (Regression)", ['k_best', 'percentile'], key="reg_mode")
if mode == 'k_best':
    param = st.sidebar.slider("Number of features (Regression)", 1, 10, 5, key="reg_param")x
if mode == 'percentile':
    param = st.sidebar.slider("Percentile (Regression)", 1, 100, 50, key="reg_param")
if mode in ['fpr', 'fdr', 'fwe']:
    param = st.sidebar.slider("Alpha (Regression)", 0.01, 0.1, 0.05, key="reg_param")

tab1, tab2 = st.tabs(['Regression','Classification'])
with tab1 :
    st.title("Feature selection for regression")
    X = load_diabetes(as_frame=True)
    score_func = st.selectbox("Score function (Regression)", ['r_regression', 'f_regression', 'mutual_info_regression'], key="reg_score_func")
    selector = GenericUnivariateSelect(score_func=eval(score_func), mode=mode, param=param)
    selector.fit(X.data, X.target)
    df = pd.DataFrame({'feature': X.data.columns, 'score': selector.scores_})
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('feature', sort='-y'),
        alt.Y('score')
    )
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart)
    with col2:
        st.write("Selected features (Regression):", selector.get_feature_names_out())

with tab2:
    X = load_iris(as_frame=True)

    score_func = st.selectbox("Score function (Classification)",
                                      ['chi2', 'f_classif', 'mutual_info_classif'], key="cls_score_func")


    selector = GenericUnivariateSelect(score_func=eval(score_func), mode=mode, param=param)
    selector.fit(X.data, X.target)
    df = pd.DataFrame({'feature': X.data.columns, 'score': selector.scores_})
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('feature', sort='-y'),
        alt.Y('score')
    )
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart)
    with col2:
        st.write("Selected features (Classification):", selector.get_feature_names_out())
