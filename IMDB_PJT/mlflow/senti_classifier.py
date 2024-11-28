##################################################################################
# [테스트 전 확인] 부분을 확인하고 테스트 하세요. (MODEL_NAME 확인 : 학습시킨 모델 폴더명으로 변경 후 테스트)
# IMDB_PJT\airflow\models 하단에 train 된 폴더가 있어야 테스트 가능
# train 된 폴더가 있다면 PKL 저장이나 로드 해볼 수 있음
##################################################################################

import pickle
import torch
import mlflow
from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSequenceClassification


############################ [테스트 전 확인] ###################################
# 모델 PKL 저장합니다. 이미 저장된 경우 필요없으면 False로 설정 
SAVE_PKL = True

# 모델 PKL 로드 : train된 model 폴더를 사용해서 한 번이라도 PKL이 저장된 경우 True
LOAD_FROM_PKL = False

# 모델 이름은 최종 학습한 모델 폴더로 설정 (폴더 naming rule 적용 필요)
# MODEL_NAME 이름을 기준으로 [[ 모델 / PKL ]] 이름 및 경로를 지정합니다.
MODEL_NAME = 'tinybert_model_test'
###################################################################################

# 모델 경로
MODLE_PATH = f'../airflow/models/{MODEL_NAME}'
PKL_PATH = f'{MODLE_PATH}_model.pkl'
PKL_TOKENIZER_PATH = f'{MODLE_PATH}_tokenizer'


# 디바이스 설정 (CPU/GPU)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path):
    # 모델 저장
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    # 토크나이저 저장
    tokenizer.save_pretrained(tokenizer_path)

# 감성 분석
def analyze_sentiment(text):

    # trained 폴더 또는 PKL로 로딩
    if LOAD_FROM_PKL:        
        # PKL 과 함께 저장 된 경로의 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(PKL_TOKENIZER_PATH, use_fast=True)
        # PKL 로드
        with open(PKL_PATH, 'rb') as f:
            model = pickle.load(f)         
    else:
        # train된 폴더의 tokenizer 사용
        tokenizer = AutoTokenizer.from_pretrained(MODLE_PATH, use_fast=True)
        # train된 폴더 model 로드
        model = AutoModelForSequenceClassification.from_pretrained(MODLE_PATH)
        # 모델과 토크나이저 저장
        if SAVE_PKL:
            save_model_and_tokenizer(model, tokenizer, PKL_PATH, PKL_TOKENIZER_PATH)
    
    classifier = pipeline(        
        'text-classification',           
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    result = classifier(text)
    return result

# main
if __name__ == "__main__":    
    
    # 분석할 텍스트
    text = "This movie was plain and boring."
    
    try:
        # 모델 로드
        print("Loading model...")
        
        # 감성 분석
        print("Analyzing sentiment...")
        result = analyze_sentiment(text)

        # mlflow 로그
        # with mlflow.start_run() as run:
        #     mlflow.log_param("model_name", MODEL_NAME)
        #     mlflow.log_metric("text", text)
        #     mlflow.log_metric("sentiment", result[0]['label'])
        #     mlflow.log_metric("confidence", result[0]['score'])
        #     print(f"Model {MODEL_NAME} logged in mlflow")      
       
        
        print(f"\nText: {text}")
        print(f"Sentiment: {result[0]['label']}")
        print(f"Confidence: {result[0]['score']:.4f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
