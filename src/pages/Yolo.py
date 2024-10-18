import streamlit as st
from ultralytics import YOLO

st.title("Yolo")

# Load a pre-trained YOLO model
model = YOLO("yolov8n.pt")