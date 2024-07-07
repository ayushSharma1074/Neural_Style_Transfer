import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv
import collections
from Neural_Style_transfer import Neural_Style_Transfer 
import time


st.set_page_config(layout="wide")

# Function to validate the form
def validate():
    error_msg=collections.defaultdict(list)
    if( style_image is None):
        error_msg["style_img"].append(" Please Upload a Style Image " )
    
    if(content_image is None):
        error_msg["content_img"].append("Please Upload a Content Image")
    return error_msg

# Callback function to update the progree bar after each epoch.
def update_progress(step):
    progress_percentage = step / st.session_state['epochs']
    text_message=f'{int(progress_percentage*100)}% complete. Please wait.'
    print(int(progress_percentage))
    if(int(progress_percentage)==1):
        text_message="Process Completed"
    progress_bar.progress(progress_percentage,text=text_message)

# Function to update the output image after each epoch.
def update_image(image):
    image=image.resize(resize_shape)
    placeholder.image(image,caption="Output Image",  use_column_width=True)
    
# Callback function to upddate both progress bar and output image passed to the Neural Style Transfer function to         
def update_image_and_progress(step, image):
    update_progress(step)
    update_image(image)
    
    
    
    
# Form
with st.form("my_form"):
    # Check if there were any validaiton errors if yes then display them below the respective field.
    val_results=st.session_state.pop("errors",{})
    style_image=st.file_uploader("Choose a Style Image", accept_multiple_files=False,type=["png","jpg"], key="style_img")
    if("style_img" in val_results ):
        for msg in val_results['style_img']:
            st.error(msg)
        
    content_image=st.file_uploader("Choose a Content Image", accept_multiple_files=False,type=["png","jpg"], key="content_img")
    
    if("content_img" in val_results ):
        for msg in val_results['content_img']:
            st.error(msg)
    
    epochs=st.slider("Total Number Of Epoch", 1,50, 10, key='epochs')
    submit=st.form_submit_button('Submit my picks')



    

        

if(submit):
    # Validate the form submission, if any error then set the message in session state and reload the page.
    error_msg=validate()
    if(len(error_msg) != 0):
        st.session_state["errors"] = error_msg
        st.rerun()
    else:
        sty_img=Image.open(style_image)
        sty_img_np=np.array(sty_img)
        content_img=Image.open(content_image)
        content_img_np=np.array(content_img)
        progress_bar = st.progress(0, text="0% complete. Please wait.")
        # Resize the images for the purpose of displaying them on the app fron page.
        resize_shape=(400,300)
        sty_img=sty_img.resize(resize_shape)
        content_img=content_img.resize(resize_shape)
        # use columns to diplay the iamge
        col1, col2, col3 = st.columns(3)   
        with col1:
            st.image(content_img,caption="Content Image", use_column_width=True)
        
        with col2:
            st.image(sty_img,caption="Style Image", use_column_width=True)
        
        with col3:
            # This acts as the placeholder, where we will update the output image after each epoch. 
            placeholder=st.empty()
            # Before optimization we set content img as the default.
            placeholder.image(content_img,caption="Content Image", use_column_width=True)
        # Run neural Style transfer on the content image using style image.         
        ouput_img=Neural_Style_Transfer(sty_img_np, content_img_np,epochs,update_image_and_progress)