import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.random.rand(200)
y = 3 * np.sin(2 * x + 1) + np.random.normal(0, 0.1, len(x))

regressor = st.selectbox('select regressor',
                         ['LR', 'KNN', 'SVM', 'Tree', 'Random Forest'])
if regressor=='LR':
    degree = st.select_slider('select degree', options=range(1, 11))
    poly = PolynomialFeatures(degree=degree)
    x_ = poly.fit_transform(x[:, None])
    rgs = LinearRegression()
if regressor=='KNN':
    k = st.select_slider('select k', options=range(1, 11))
    rgs = KNeighborsRegressor(n_neighbors=k)
if regressor=='SVM':
    C = st.select_slider('select C', options=[0.1, 1, 10, 100])
    kernel = st.selectbox('select kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    rgs = SVR(C=C, kernel=kernel)
if regressor=='Tree':
    depth = st.select_slider('select depth', options=range(1, 11))
    rgs = DecisionTreeRegressor(max_depth=depth)
if regressor=='Random Forest':
    n_estimators = st.select_slider('select n_estimators', options=[10, 50, 100, 200])
    rgs = RandomForestRegressor(n_estimators=n_estimators)

if regressor=='LR':
    rgs.fit(x_, y)
    z = rgs.predict(x_)
else:
    rgs.fit(x[:, None], y)
    z = rgs.predict(x[:, None])

plt.figure()
plt.plot(x, y, '.')
plt.plot(x, z, '.r')

st.pyplot(plt)