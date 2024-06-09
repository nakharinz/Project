import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from imdb import Cinemagoer

ia = Cinemagoer()

#ratings_path = 'https://raw.githubusercontent.com/smanihwr/ml-latest-small/master/ratings.csv'

@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/smanihwr/ml-latest-small/master/movies.csv')
    df = df.join(df.pop('genres').str.get_dummies('|'))
    return df

if 'movies' not in st.session_state:
    st.session_state['movies'] = load_data()

def get_cover_url(id):
    print(id)
    links = pd.read_csv('links.csv')  # SQL connection
    idx = links['movieId'] == id
    if 'cover' not in links.columns or links.loc[idx, 'cover'].isna().values[0]:
        mov = ia.get_movie(links[idx]['imdbId'].values[0])
        links.loc[idx, 'cover'] = mov['cover']
        links.to_csv('links.csv', index=False)  # update SQL database
    cover = links.loc[idx, 'cover'].values[0]
    return cover

show_cover = st.sidebar.checkbox('Show cover', value=False)
movies = st.session_state['movies']
x = []
for i in movies.columns[3:]:
    x.append(st.sidebar.slider(i, 0, 5, 0))
x = np.array(x)
user_profile = x / x.sum()
rank = (movies.iloc[:, 3:] * user_profile).sum(axis=1)

st.header('All movies in datasets')
st.write(movies)
p = movies.iloc[:, 3:].sum(axis=0)
fig = px.pie(p, names=p.index, values=p.values)
st.plotly_chart(fig)

fig = px.bar(p, x=p.values, y=p.index)
st.plotly_chart(fig)

fig = px.imshow(movies.iloc[:, 3:].corr().round(1), text_auto=True)
st.plotly_chart(fig)


st.header('Recommendation')
# st.write(user_profile)

df = pd.DataFrame(columns=['movieId', 'title', 'rank'])
df['movieId'] = movies['movieId']
df['title'] = movies['title']
df['rank'] = rank
df = df.sort_values('rank', ascending=False)
st.write(df)

if show_cover:
    for i, row in df.iloc[:10, :].iterrows():
        print(row['movieId'])
        st.image(get_cover_url(row['movieId']))
        st.write(row['title'], row['rank'])
        st.write('---')