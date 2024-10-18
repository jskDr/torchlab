import requests
import streamlit as st
import torch
from PIL import Image
from io import BytesIO

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
im_url = "https://ultralytics.com/images/zidane.jpg"

# Download and display the image
response = requests.get(im_url)
image = Image.open(BytesIO(response.content))
st.image(image, caption="Original Image", use_column_width=True)

results = model(image)

# Display detection results
st.write("Detection Results:")
st.write(results.pandas().xyxy[0])