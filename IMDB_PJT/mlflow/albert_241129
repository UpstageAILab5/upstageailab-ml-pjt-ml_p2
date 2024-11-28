# 라이브러리 설정
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# mlflow 라이브러리
import mlflow
import mlflow.pytorch

from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# 모델 관련 라이브러리
import evaluate
from evaluate import load
import torch 
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

# transformers 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, Trainer, TrainingArguments, pipeline   

# # airflow 라이브러리
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

# ----------------------------------------------------------------

# 환경 변수 설정
os.environ['NO_PROXY'] = '*'  # airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
# 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'


# 현재 경로 설정
current_path = os.getcwd()

# nltk 데이터 경로 설정
nltk_data_path = os.path.join(current_path, 'nltk_data')
nltk.data.path.append(nltk_data_path)

try: 
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
    nltk.download('maxent_ne_chunker', download_dir=nltk_data_path)
    nltk.download('words', download_dir=nltk_data_path)
    print('NLTK 데이터 다운로드 완료')
except Exception as e:
    print(f'NLTK 데이터 다운로드 중 오류 발생: {e}')

# 토크나이저 경로 설정
tokenizer_path = os.path.join(current_path, 'tokenizer')
# 데이터 경로 설정
data_path = os.path.join(current_path, 'data', 'IMDB_Dataset.csv')
dataset_path = os.path.join(current_path, 'data', 'tk_dataset')
# output_path = os.path.join(current_path, 'train_dir')
model_path = 'albert_lemmi'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


# 모델 준비
model_name = 'albert-base-v2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# mlflow 설정
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('IMDB_ALBERT_1128')

# ------------------------------------------------------------

# Airflow 기본 설정
default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 11, 27),
    'retries': 1,
    'env': {
        'NO_PROXY': '*',   # airflow로 외부 요청할 때 
        'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'   # 파이썬 버전 이슈 해결
    }
}

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer

def compute_accuracy(predictions):
    predict = np.argmax(predictions.predictions, axis=1)
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=predict, 
                           references=predictions.label_ids)


# def compute_accuracy(predictions):
#     logits = predictions.predictions
#     labels = predictions.label_ids
#     pred_labels = np.argmax(logits, axis=1)
#     accuracy = evaluate.load('accuracy')
#     return accuracy.compute(predictions=pred_labels, 
#                            references=labels)

# 데이터 로드 함수 정의
def data_load():
    # 데이터 로드
    df = pd.read_csv(data_path)
    print("Available columns:", df.columns.tolist())
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    print(f'데이터 로드 완료')
    # 라벨 딕셔너리 생성

    return dataset


def data_preprocess(dataset):
    def clean_text_function(examples):
        clean_text = []
        lemmatizer = WordNetLemmatizer()

        for text in examples['review']:
            try:
                text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
                text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
                text = re.sub(r'\d+', '', text)  # 숫자 제거
                text = text.lower()  # 소문자로 변환
                text = text.strip()  # 문자열 양쪽 공백 제거
                text = text.replace('br', '')  # 'br' 태그 제거

                words = word_tokenize(text)
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                text = ' '.join(lemmatized_words)
                
                clean_text.append(text)
            except Exception as e:
                print(f'데이터 전처리 중 오류 발생: {e}')
        return {'review': clean_text}
    
    dataset_clean = {}
    for split in dataset: #split: train, test
        dataset_clean[split] = dataset[split].map(clean_text_function, batched=True)
    print(f'텍스트 정제 완료')

    
    # 토크나이저 적용
    tokenizer = load_tokenizer()
    dataset_tokenized = {}
    for split in dataset_clean: #split: train, test
        dataset_tokenized[split] = dataset_clean[split].map(
            lambda x: tokenizer(
                x['review'], # 토크나이저 적용할 데이터
                padding='max_length', # 최대 길이 이상 데이터는 잘라냄
                truncation=True, # 최대 길이 이상 데이터는 잘라냄
                max_length=300, # 최대 길이 설정
                return_tensors=None # mlflow만 사용할 시 None으로,  'pt'는 pytorch tensor로 반환하는 것을 의미
            ), 
            batched=True
        )

    # 라벨 적용
    label2id = {'positive': 1, 'negative': 0}
    dataset_labeled = {}
    for split in dataset_tokenized: #split: train, test
        dataset_labeled[split] = dataset_tokenized[split].map(
            lambda x: {'label': label2id[x['sentiment']]}
        )
    dataset_labeled = DatasetDict(dataset_labeled)
    dataset_saved_path = os.path.join(dataset_path, f'processed_dataset_{timestamp}.json')
    dataset_labeled.save_to_disk(dataset_saved_path)
    print(f'데이터 전처리 완료')
    return dataset_labeled, label2id, dataset_saved_path

# 모델 학습 및 평가 함수 정의
def train_evaluate_model(dataset_labeled):
    # ti = kwargs['ti']
    # os.makedirs(output_path, exist_ok=True)
    
    # 라벨 설정. label2id 딕셔너리 키 값을 뒤집음
    id2label = {0: 'negative', 1: 'positive'}
    # 모델 설정
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    # 학습 설정
    args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        eval_strategy='epoch'
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_labeled['train'],
        eval_dataset=dataset_labeled['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy    
    )

    mlflow.autolog()
    # mlflow에 모델 저장
    with mlflow.start_run():
        trainer.train()
        print(f'모델 학습 완료')

        evaluation = trainer.evaluate()
        predictions = trainer.predict(dataset_labeled['test'])
        accuracy_score = compute_accuracy(predictions) 
        evaluation_result = {
            "model_name": model_name,
            "eval_loss": evaluation['eval_loss'],  # 평가 손실
            "eval_accuracy": evaluation['eval_accuracy'],  # 평가 정확도
            "predict_accuracy": accuracy_score['accuracy'],  # 예측 정확도
            "eval_runtime": evaluation['eval_runtime']  # 평가 실행 시간
        }   

        print(f'모델 평가 완료')
        mlflow.log_metrics({"test_accuracy": accuracy_score['accuracy']})
        print(f'모델 저장 완료')
        trainer.save_model(model_path)
        mlflow.pytorch.log_model(model, 
                                 run_name=f'{model_name}_{timestamp}',
                                 artifact_path=model_path,
                                 registered_model_name='albert_imdb',
                                 tags={'dataset': 'IMDB', 
                                       'model': 'albert', 
                                       'timestamp': timestamp})
        model_save_pkl = os.path.join(model_path, f'model_{timestamp}.pkl') # pkl로도 저장
        joblib.dump(model, model_save_pkl) 
        print(f'모델 저장 완료: {model_save_pkl}')
        
    print(f'모델 학습 및 평가 완료')
    return model, model_path, evaluation_result


def slack_notification(evaluation_result, run_name, dataset_path):
    message = f"""
*Model Training Completed*
• Dataset: {os.path.basename(dataset_path)}
• Model: {os.path.basename(run_name)}
• Model Name: {evaluation_result['model_name']}
• Training Results:
  - Test Loss: {evaluation_result['eval_loss']:.4f}
  - Test Accuracy: {evaluation_result['eval_accuracy']:.4f}
  - Predict Accuracy: {evaluation_result['predict_accuracy']:.4f}
• Training Duration: {evaluation_result['eval_runtime']:.2f} seconds
    """
    slack_webhook_url = '**'
    payload = {
        'text': message
    }
    response = requests.post(slack_webhook_url, json=payload)
    print(f'슬랙 알림 완료')


if __name__ == "__main__":

    dataset = data_load()
    dataset_labeled, label2id, dataset_saved_path = data_preprocess(dataset)
    model, run_name, evaluation_result = train_evaluate_model(dataset_labeled)
    slack_notification(evaluation_result, run_name, dataset_path)    
