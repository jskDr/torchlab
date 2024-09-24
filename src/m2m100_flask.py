"""
Model Execution
$ python app.py
"""
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 모델과 토크나이저 로드
model_name = "JamesKim/m2m100-ft3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/translate', methods=['POST'])
def translate():
    # JSON에서 입력 텍스트 가져오기
    data = request.get_json()
    text = data.get('text', '')

    # 입력 텍스트를 토큰화
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 모델을 사용하여 번역 생성
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 번역된 텍스트를 JSON 형식으로 반환
    return jsonify({'translation': translation})

if __name__ == '__main__':
    # Flask 애플리케이션 실행
    app.run(host='0.0.0.0', port=5000)
