# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# 라이브러리 설정
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from datasets import load_dataset, Dataset, load_from_disk
import joblib

# 평가 지표 라이브러리
import evaluate
from evaluate import load
import torch
import torch.nn.functional as F
# mlflow 라이브러리
import mlflow
import mlflow.pytorch

# transformers 라이브러리
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
from transformers import Trainer, TrainingArguments, pipeline   
# from gluonnlp import nlp, Vocab
from kobert_tokenizer import KoBERTTokenizer

# # airflow 라이브러리
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator


# 환경 변수 설정
os.environ['NO_PROXY'] = '*'  # airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
# 모델 준비
model_name = 'skt/kobert-base-v1'   # 또는 monologg/kobert
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 경로 설정
# current_path = os.getcwd()
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data', 'ratings.txt')
dataset_path = os.path.join(current_path, 'data', 'tk_dataset')
output_path = os.path.join(current_path, 'train_dir')
model_path = os.path.join(current_path, 'kobert_saved')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

mlflow.set_tracking_uri('http://127.0.0.1:5005')
mlflow.set_experiment('IMDB_Kobert_Model_1128')


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
    tokenizer = KoBERTTokenizer.from_pretrained(
        model_name, 
        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
    # tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False) # 토크나이저 설정
    return tokenizer

def compute_accuracy(predictions):
    logits = predictions.predictions
    labels = predictions.label_ids
    pred_labels = np.argmax(logits, axis=1)
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=pred_labels, 
                           references=labels)

# 데이터 전처리 함수 정의
def prepare_data():
    # 데이터 로드
    df = pd.read_csv(data_path, sep='\t')
    print("Columns in DataFrame:", df.columns)  # Check the columns
    print("Data Types:", df.dtypes)  # Check data types
    print("First few rows of the DataFrame:\n", df.head())  # Display first few rows
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    print(f'데이터 로드 완료')
    
    # 라벨 딕셔너리 생성
    label2id = {'1': 1, '0': 0}
    
    # 토크나이저 적용
    tokenizer = load_tokenizer()
    
    def tokenize_function(examples):
        texts = [str(doc) for doc in examples['document']]
        tokenized = tokenizer(
            texts,  # Access review column from examples
            padding='max_length',
            truncation=True,
            max_length=300,
            return_tensors=None  # Change to None when using batched=True
        )
        # Add labels
        tokenized['labels'] = [label2id[str(label)] for label in examples['label']]
        return tokenized
    
    # Apply tokenization using map
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    dataset.save_to_disk(dataset_path)
    print(f'데이터 전처리 완료')
    return dataset, label2id, dataset_path

class CustomBertClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
        return torch.nn.functional.log_softmax(logits, dim=1) if loss is None else (loss, logits)
    
# 모델 학습 및 평가 함수 정의
def train_eval_model(dataset, label2id):
    os.makedirs(output_path, exist_ok=True)
    # 라벨 설정. label2id 딕셔너리 키 값을 뒤집음
    id2label = {0: '0', 1: '1'}
    # 모델 설정
    model = CustomBertClassifier(
        model_name,
        num_labels=len(label2id)
    ).to(device)
    # 학습 설정
    args = TrainingArguments(
        output_dir=output_path,
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
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy    
    )
    
    # trainer.save_model(model_path)


    mlflow.autolog()
    # mlflow에 모델 저장
    with mlflow.start_run():
        trainer.train()
        print(f'모델 학습 완료')

        evaluation = trainer.evaluate()
        predictions = trainer.predict(dataset['test'])
        accuracy_score = compute_accuracy(predictions[:2])
        evaluation_result = {
            "eval_loss": evaluation['eval_loss'],  # 평가 손실
            "eval_accuracy": evaluation['eval_accuracy'],  # 평가 정확도
            "predict_accuracy": accuracy_score['accuracy'],  # 예측 정확도
            "eval_runtime": evaluation['eval_runtime']  # 평가 실행 시간
        }
        print(f'모델 평가 완료')
        mlflow.log_metrics(evaluation_result)
        print(f'모델 저장 완료')
        trainer.save_model(model_path)
        mlflow.pytorch.log_model(model, "model")
        model_save_pkl = os.path.join(model_path, f'model_{timestamp}.pkl')
        joblib.dump(model, model_save_pkl) 
        print(f'모델 저장 완료')
        
    print(f'모델 학습 및 평가 완료')
    return model, model_path, evaluation_result



def slack_notification(evaluation_result, model_path, data_path):
    message = f"""
*Model Training Completed*
• Dataset: {os.path.basename(data_path)}
• Model: {os.path.basename(model_path)}
• Training Results:
  - Eval Loss: {evaluation_result['eval_loss']:.4f}
  - Eval Accuracy: {evaluation_result['eval_accuracy']:.4f}
  - Predict Accuracy: {evaluation_result['predict_accuracy']:.4f}
• Training Duration: {evaluation_result['eval_runtime']:.2f} seconds
• Samples Per Second: {evaluation_result['eval_runtime'] / evaluation_result['predict_accuracy']:.2f}
    """
    slack_webhook_url = 'https://hooks.slack.com/services/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'
    payload = {
        'text': message
    }
    response = requests.post(slack_webhook_url, json=payload)
    print(f'슬랙 알림 완료')


if __name__ == "__main__":

    dataset, label2id, dataset_path = prepare_data()
    model, model_path, evaluation_result = train_eval_model(dataset, label2id)
    slack_notification(evaluation_result, model_path, dataset_path)    