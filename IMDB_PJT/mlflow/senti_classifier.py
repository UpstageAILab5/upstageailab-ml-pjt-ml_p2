## 아래 model_path에 저장한 모델을 로드한 후에 실행

import torch
import joblib
import mlflow
from transformers import pipeline, AutoTokenizer

# 모델 이름은 학습에 사용한 모델 이름과 동일하게 설정
model_name = 'huawei-noah/TinyBERT_General_4L_312D'

# 디바이스 설정 (CPU/GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 모델 로드
def load_saved_model(model_path):
    model = joblib.load(model_path)
    return model

# 감성 분석
def analyze_sentiment(text, model):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    classifier = pipeline(
        task='sentiment-analysis',
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    result = classifier(text)
    return result

# 예시 사용
if __name__ == "__main__":
    # 실제 모델 경로로 변경
    model_path = "tinybert_saved/model_20241127_142502.pkl"  # 실제 모델 경로로 이름 변경
    
    # 분석할 텍스트
    text = "This movie was plain and boring."
    
    try:
        # 모델 로드
        print("Loading model...")
        model = load_saved_model(model_path)
        
        # 감성 분석
        print("Analyzing sentiment...")
        result = analyze_sentiment(text, model)
        
        print(f"\nText: {text}")
        print(f"Sentiment: {result[0]['label']}")
        print(f"Confidence: {result[0]['score']:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")