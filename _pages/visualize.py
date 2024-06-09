import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('Datasets')
ds = st.selectbox('Select Dataset',
                  sns.get_dataset_names())
df = sns.load_dataset(ds)
corr = df.corr(numeric_only=True)
fig = plt.figure()
sns.heatmap(corr, annot=corr)

fig

lat = np.random.randint(130000, 150000, 100) / 10000
lon = np.random.randint(1005000, 1010000, 100) / 10000
df_num = pd.DataFrame(np.vstack([lat, lon]).T,
                      columns=['lat', 'lon'])

st.area_chart(df_num)

st.map(df_num)

st.bar_chart(df_num)

st.line_chart(df_num)

st.scatter_chart(df_num)

