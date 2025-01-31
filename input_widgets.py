import streamlit as st
from PIL import Image, ImageColor
import numpy as np
name = st.text_input('Enter your name')
st.write('Hello', name)
age = st.number_input('Enter your age', step=1,
                      min_value=0, max_value=130)
st.write('Your age is', age)
bio = st.text_area('Enter your bio', height=200)
birthday = st.date_input('Enter your birthday')
st.write(birthday)
time = st.time_input('What time did you wake up?')
st.write(time)
avatar = st.camera_input('Cheese!!')
if avatar is not None:
    avatar = np.array(Image.open(avatar))[::-1, :, :]
    st.image(avatar)
    Image.fromarray(avatar).save('avatar_server.png')
    st.download_button('Download image',
                       open('avatar_server.png', 'rb'),
                       'avatar_client.png')
st.write('server image')
img = st.file_uploader('Upload image', type=['jpg', 'png'])
if img is not None:
    st.image(Image.open(img))

clicked = st.button('Click me!')
if clicked:
    st.write('Button clicked!')

checked = st.checkbox('Check me!')
if checked:
    st.write('Checkbox checked!')

choice = st.radio('Choose one', ['A', 'B', 'C'])
st.write('You chose', choice)

mul_choices = st.multiselect('Choose many', ['A', 'B', 'C'])
st.write('You chose', mul_choices)

rating = st.slider('Rate me', 1, 5, 1)
st.write('You rated', rating)

level = st.select_slider('Select level', options=['Beginner', 'Intermediate', 'Advanced'])
st.write('You selected', level)

color = st.color_picker('Pick a color')
st.write(f'You picked {ImageColor.getcolor(color, "RGB")}')