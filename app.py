import streamlit as st
import requests as req
import io
import tensorflow as tf
import tensorflow_hub as hub
import time
import numpy as np
from textblob import TextBlob
from PIL import Image, ImageEnhance, ImageOps, ImageDraw


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

def object_detection():
    # Isi dengan fungsionalitas image object detection
    st.write("Image Object Detection tab content.")

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image

def run_detector(detector, img):
  img_array = np.array(img)
  converted_img = tf.image.convert_image_dtype(img_array, tf.float32)[tf.newaxis, ...]
  start_time = time.time()
  result = detector(converted_img)
  end_time = time.time()

  result = {key: value.numpy() for key, value in result.items()}

  print("Found %d objects." % len(result["detection_scores"]))
  print("Inference time: ", end_time - start_time)

  image_with_boxes = draw_boxes(
    img_array, result["detection_boxes"],
    result["detection_class_entities"], result["detection_scores"])

  return image_with_boxes

def main():
    st.title('Python Processing Playground')

    # # Membuat tiga tab
    # st.sidebar.title("Navigation")
    # tabs = ["Enhance Image", "Sentiment Analysis", "Image Object Detection"]
    # # selected_tab = st.sidebar.selectbox("Select Tab", tabs)
    # selected_tab = st.sidebar.radio("Select Tab", tabs)
    container = st.container(border=True)
    enhance_button, sentiment_button, object_detection_button = container.tabs(["Enhance Image", "Sentiment Analysis", "Image Object Detection"])

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
        response = req.get(news_url)
        if response != False :
            response = response.json()
            sentiment_analysis(response)
        else :
            print("cannot get API")

    # Tab Image Object Detection
    with object_detection_button:
        st.header("Object Detection Image")
        st.write('Upload an image and Detect object from the image')

        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="enhance_object")

        if uploaded_image is not None:
            img = load_img(uploaded_image)
            st.image(img, caption='Original Image', use_column_width=True)
            # enhancement_factor = st.slider('Enhancement Factor', 0.0, 2.0, 1.0)

            if st.button('Enhance'):
                module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
                detector = hub.load(module_handle).signatures['default']

                result_image = run_detector(detector, img)
                buffered = io.BytesIO()
                result_image_pil = Image.fromarray(result_image)
                result_image_pil.save(buffered, format="JPEG")
                # image = Image.open(uploaded_image)
                # enhanced_image = enhance_image(image, enhancement_factor)
                st.image(result_image_pil, caption='Object Detection', use_column_width=True)

if __name__ == "__main__":
    main()
