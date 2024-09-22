from typing_extensions import TypedDict

import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFile
import torch
import pandas as pd

class ClassificationResult(TypedDict):
    label: str
    prob: float

def classify(image:ImageFile):
    # 모델과 프로세서 로드
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 분류하고자 하는 클래스 레이블 정의
    candidate_labels = ["a photo of a lion", "a photo of a dog", "a photo of a horse", "a photo of a bear"]

    # 이미지와 텍스트 전처리
    inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # 이미지-텍스트 유사도 계산
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # 결과 출력
    st.header("Classification Result")
    
    for i, label in enumerate(candidate_labels):
        st.write(f"{label}: {probs[0][i].item():.2%}")
        print(f"{label}: {probs[0][i].item():.2%}")

    # 가장 높은 확률의 클래스 출력
    st.write(f"\nBest match: {candidate_labels[probs.argmax().item()]}")
    print(f"\nBest match: {candidate_labels[probs.argmax().item()]}")

def main():
    st.set_page_config(
        page_title='CLIP by PyTorch & Transformers',
        layout='centered',
        page_icon='❤️'
    )
    st.title('CLIP')
    # 이미지 로드 (여기서는 'image.jpg'라고 가정)
    image = Image.open("./images/image.jpg")
    st.image(image)
    classify(image)
    
if __name__ == '__main__':
    main()