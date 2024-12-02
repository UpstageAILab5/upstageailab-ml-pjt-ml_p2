# 라이브러리 설정
import os
import pandas as pd
import numpy as np
from datetime import datetime
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

# 평가 지표 라이브러리
import evaluate
from evaluate import load
import torch
import joblib

# mlflow 라이브러리
import mlflow
import mlflow.pytorch

# transformers 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, pipeline   

# # airflow 라이브러리
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator


# 환경 변수 설정
os.environ['NO_PROXY'] = '*'  # airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
# 모델 준비
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 모델 이름 설정
model_name = 'huawei-noah/TinyBERT_General_4L_312D'

# 데이터 경로 설정
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data', 'IMDB_Dataset.csv')
dataset_path = os.path.join(current_path, 'data', 'tk_dataset')
output_path = os.path.join(current_path, 'train_dir')
model_path = os.path.join(current_path, 'tinybert_saved')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

sentiment_text = 'I love this movie!'

default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 11, 22),
    'retries': 1,
}

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer

def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=predictions, 
                           references=labels)


def prepare_data():
# 데이터 로드
    df = pd.read_csv(data_path, index_col=0)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)

# 라벨 설정
    label2id = {'positive': 1, 'negative': 0}

# 토크나이저 적용
    tokenizer = load_tokenizer()
    dataset = dataset.map(
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
    dataset = dataset.map(lambda x: {'label': label2id[x['sentiment']]})
    
    return dataset, label2id

def train_model(label2id, dataset):
    # 라벨 설정
    id2label = {0: 'negative', 1: 'positive'}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        learning_rate=2e-5,
        evaluation_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy
    )
    train_result = trainer.train()

# mlflow에 모델 저장
    mlflow.pytorch.log_model(model, model_name)  # mlflow에 모델 저장. pytorch 모델 형식으로 저장
    mlflow.log_param("model_name", model_name)  # mlflow에 모델 이름 저장

    model_saved_path = os.path.join(model_path, f'model_{timestamp}.pkl') # 모델 저장 경로: 모델 저장 경로/모델 이름.pkl
    os.makedirs(model_path, exist_ok=True)  # 모델 저장 경로 생성
    joblib.dump(model, model_saved_path)  # 모델 저장. joblib: 파이썬 객체를 파일로 저장하는 라이브러리
    return model, model_saved_path 

def evaluate_model(model, dataset):
    trainer = Trainer(
        model=model,
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy
    )
    # 모델 평가
    evaluation_result = trainer.evaluate(dataset['test'])

    mlflow.log_metrics({
        "test_loss": evaluation_result['eval_loss'],
        "test_accuracy": evaluation_result['eval_accuracy']
    })
    return evaluation_result




if __name__ == "__main__":
    mlflow.autolog()
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('IMDB_Model_Training')
    
    dataset, label2id = prepare_data()
    print("데이터 준비 완료")

    with mlflow.start_run(run_name=f"tinybert_{timestamp}"):
        
        # train_model 함수 호출
        model, model_path = train_model(label2id, dataset)
        print("모델 학습 완료: {model_path}")

        # evaluate_model 함수 호출
        evaluation_result = evaluate_model(model, dataset)
        mlflow.log_metrics(evaluation_result)
        print(f"모델 평가 완료: {evaluation_result}")


