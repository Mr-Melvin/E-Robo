import streamlit as st
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from streamlit_js_eval import streamlit_js_eval
from keras.models import load_model 


def prediction(model,photo):
    img = Image.open(photo)
    img_arr = np.array(img)
    img = cv2.resize(img_arr,(224,224))
    img = img.reshape(1,224,224,3)

    result = model.predict(img)
    ind = result.argmax(axis=1)

    display_img = cv2.resize(img_arr,(350,250))

    if ind == 0:
        return 'African Elephant',display_img
    elif ind == 1:
        return 'Asian Elephant',display_img
    else:
        return 'Something went wrong'
    


try:
    model = load_model('cnn_model/elephant3_vgg_78.h5',compile=False)


    st.set_page_config(page_title='E-Robo',layout='wide',initial_sidebar_state="collapsed")


    col1,col2,col3 = st.columns([1,4,1])
    with col2:
        st.title("Hi, I'm E-Robo and here to predict whether the elephant is African or Asian.")

    col4, col5, col6 = st.columns([1,7,1])
    with col5:
        image = Image.open('images/robo.png')
        st.image(image,width=150)
        st.write(' ')
        uploaded_files = st.file_uploader(label=':red[WARNING : ] Upload only elephant images of format .png,jpg or jpeg .',
                                                accept_multiple_files=True,
                                                type=['png','jpg','jpeg'],
                                                ) 
            
    st.write(" ")
    button_col1, button_col2, button_col3 = st.columns([3,3,3])
    with button_col2:
        pred = st.button('Predict',use_container_width=True,type='primary')

    st.write('----')

    if pred:
        if uploaded_files == []:
            st.error("Please upload elephant photo.")    
        else:
            st.title(':green[E-Robo Says:]')
            st.write(" ")
            group = []
            for i in range(0,len(uploaded_files),4):
                group.append(uploaded_files[i:i+4]) 
                    
            for grp in group:
                col = st.columns(4)
                for i,pic in enumerate(grp):
                    r,img = prediction(model,pic)
                    col[i].write(r)
                    col[i].image(img)
                    st.write(" ")

            button_col4, button_col5, button_col6 = st.columns([3,3,3]) 
            with button_col5:               
                if st.button("Close Result",use_container_width=True,type="secondary"):
                    streamlit_js_eval(js_expressions="parent.window.location.reload()")

except ValueError:
    st.info("something went wrong")
    if st.button("Try Again"):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")