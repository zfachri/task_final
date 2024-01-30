import streamlit as st
import requests as req
import io
import tensorflow as tf
import tensorflow_hub as hub
import time
import os # used to save images in a directory
import cv2
import numpy as np
import object_detection as detect 
from textblob import TextBlob
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageColor, ImageFont



def enhance_image(image, enhancement_factor):
    enhanced_image = ImageEnhance.Contrast(image).enhance(enhancement_factor)
    return enhanced_image

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    return 'Positif' if polarity > 0 else ('Negatif' if polarity < 0 else 'Netral')

def sentiment_analysis(data):
    # Isi dengan fungsionalitas sentiment analysis
    st.write("Sentiment Analysis tab content.\n")

    news_text = ''
    for article in data['articles']:
        news_text += article['title'] + ' '

# Simpan teks berita ke file teks
    with open('news_text.txt', 'w') as file:
        file.write(news_text)
    with open('news_text.txt', 'r', encoding='utf-8') as file:
        readFile = file.read()

    # Analisis sentimen
    sentiment_result = analyze_sentiment(readFile)

    sentiment_result = str(sentiment_result)

    if sentiment_result.lower() == "positif"  :
        warna = "green"
    else :
        warna = "red"

    # Output hasil analisis sentimen
    # st.markdown(f'<p style="color:{warna};">{teks}</p>', unsafe_allow_html=True)
    result_text = f'<span style="color:{warna};">{sentiment_result}</span>'
    st.write("Sentiment Analysis Result: "+ result_text,unsafe_allow_html=True)

def load_img(path):
  img = Image.open(path)
  img = img.convert("RGB")
  img = ImageOps.fit(img, (500, 280), Image.LANCZOS)
  return img

def multiscale_retinex(image_path, sigma_list=[100, 200, 300]):

    image = np.array(image_path) 
    retinex_image = np.zeros_like(image, dtype=np.float32)

    for sigma in sigma_list:
        log_image = np.log1p(image.astype(np.float32))
        gaussian = cv2.GaussianBlur(log_image, (0, 0), sigma)
        retinex_image += log_image - gaussian

    retinex_image /= len(sigma_list)
    retinex_image = np.exp(retinex_image) - 1.0

    # Normalisasi gambar
    retinex_image_normalized = (retinex_image - np.min(retinex_image)) / (np.max(retinex_image) - np.min(retinex_image)) * 255
    retinex_image_normalized = retinex_image_normalized.astype(np.uint8)

    enhance_image = cv2.cvtColor(retinex_image_normalized, cv2.COLOR_BGR2RGB)
    return enhance_image

def manipulate_image(input_path):
    # image = Image.open(input_path)
      # Convert the image to grayscale
    gray_image = input_path.convert("L")
    # Lakukan manipulasi lainnya, seperti enhancement MSRCR
    # Misalnya, menggunakan ImageEnhance untuk meningkatkan kecerahan
    enhancer = ImageEnhance.Brightness(gray_image)
    enhanced_image = enhancer.enhance(2.0)  # Contoh: meningkatkan kecerahan sebesar 2 kali
    # Lakukan manipulasi lainnya, seperti enhancement MSRCR
    return enhanced_image


def main():
    st.title('Python Processing Playground')

    # # Membuat tiga tab
    # st.sidebar.title("Navigation")
    # tabs = ["Enhance Image", "Sentiment Analysis", "Image Object Detection"]
    # # selected_tab = st.sidebar.selectbox("Select Tab", tabs)
    # selected_tab = st.sidebar.radio("Select Tab", tabs)
    container = st.container(border=True)
    enhance_button, multi_retinex_button,sentiment_button, object_detection_button = container.tabs(["Enhance Image","Multi RetinexScale Image", "Sentiment Analysis", "Image Object Detection"])

    with enhance_button :
    # Tab Enhance Image
        st.header("Enhance Your Images")
        st.write('Upload an image and enhance it!')

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="enhance_image")

        if uploaded_image is not None:
            st.image(uploaded_image, caption='Original Image', use_column_width=True)

            enhancement_factor = st.slider('Enhancement Factor', 0.0, 2.0, 1.0)

            if st.button('Enhance'):
                image = Image.open(uploaded_image)
                enhanced_image = enhance_image(image, enhancement_factor)
                st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)
    
    with sentiment_button :
    # Tab Sentiment Analysis
        st.header("Sentiment Analysis")
        news_url = "https://newsapi.org/v2/top-headlines?country=id&category=technology&apiKey=e1c7f6efec974dddb22076143d05bfe8";
        try :
          response = req.get(news_url)
          response = response.json()
          if response['status'] != "error" :
            sentiment_analysis(response)
          else :
            st.write("cannot get API")
        except req.exceptions.RequestException as e:
            # Handle any exception that occurred during the request
            st.warning(f"Error during request: {e}")

    with multi_retinex_button:
       st.header("Multi RetinexScale Image")
       st.write('Upload an image and see the enhancement image using gaussian blur')

       uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="gaussian_blur")

       if uploaded_image is not None:
        st.image(uploaded_image, caption='Original Image', use_column_width=True)

        if st.button("MultiRetinex Scale") :
          input_image = Image.open(uploaded_image)
          grey_scale = manipulate_image(input_image)
          # grey_scale_display = Image.open(io.BytesIO(grey_scale))

          multi_scale_retinex = multiscale_retinex(input_image)
          # multi_scale_retinex_display = Image.Open(io.BytesIO(multi_scale_retinex))

          st.image([grey_scale,multi_scale_retinex], caption=["grey Scale Image","Multi Retinex Scale Image"], use_column_width=True)

    # Tab Image Object Detection
    with object_detection_button:
        labels, colors, height, width, interpreter = detect.define_tf_lite_model()
        st.header("Object Detection Image")
        st.write('Upload an image and Detect object from the image')
        st.write('Note: This model only take some object some result may be not really accurate')

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="enhance_object")

        if uploaded_image is not None:
            img = load_img(uploaded_image)
            st.image(img, caption='Original Image', use_column_width=True)
            # enhancement_factor = st.slider('Enhancement Factor', 0.0, 2.0, 1.0)
            with open(os.path.join("tempDir",uploaded_image.name),"wb") as f: 
                f.write(uploaded_image.getbuffer())  

            if st.button('Object Detection'):
                resultant_image = detect.display_results(labels, 
                                         colors, 
                                         height, 
                                         width,
                                         "tempDir/" + uploaded_image.name, 
                                         interpreter, 
                                         threshold=0.5)
                st.image(Image.fromarray(resultant_image), use_column_width=True)

                file_path = os.path.join("tempDir", uploaded_image.name)
                os.remove(file_path)
                # result_image= object_detection(input_image)
                # # buffered = io.BytesIO()
                # result_image_pil = Image.fromarray(result_image)
                # # result_image_pil.save(buffered, format="JPEG")
                # # image = Image.open(uploaded_image)
                # # enhanced_image = enhance_image(image, enhancement_factor)
                # st.image(result_image_pil, caption='Object Detection', use_column_width=True)
    
if __name__ == "__main__":
    main()
