# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
############### ------------------------------라이브러리 설정------------------------------ ###############
#----------------------------------------------------------------------------------------------------#
# 기본 라이브러리
import pandas as pd                 # 기본 데이터 처리
import numpy as np                  # 숫자 처리
import os                           # 시스템 경로 설정
import time                         # 시간 설정
import pytz                         # 시간대 설정
from datetime import datetime       # 시간 설정
import json

# 데이터 처리
from datasets import Dataset,  DatasetDict                          # 데이터 처리
import re

# MLflow 라이브러리
import mlflow                                   # mlflow 설정
import mlflow.pytorch                           # mlflow 설정. pytorch 모델 저장

# 트랜스포머 처리
from transformers import BertModel
import torch                                                            # 텐서 처리
import torch.nn.functional as F                                         # 텐서 처리. F: 텐서 함수 처리

# Airflow 라이브러리
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator


############### ------------------------------기본 환경 설정------------------------------ ###############
#----------------------------------------------------------------------------------------------------#

# 환경 변수 설정
os.environ['NO_PROXY'] = '*'                                    # 환경 변수 설정 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요

# 경로 설정
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data', 'ratings.txt')
tokenizer_path = os.path.join(current_path, 'kobert_token')

# 모델 설정
model_name = 'skt/kobert-base-v1'   # 또는 monologg/kobert
sample_size = 10
model_path = 'kobert_model'

############### ------------------------------코드 작성 ----------------------------- ###############
#-------------------------------------------------------------------------------------------------#

# 토크나이저 로드
def load_tokenizer():
    # 모델 토크나이저 처리
    from kobert_tokenizer import KoBERTTokenizer

    tokenizer = KoBERTTokenizer.from_pretrained(
        model_name, 
        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
    # tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False) # 토크나이저 설정
    return tokenizer

# 평가 지표 설정
def compute_accuracy(predictions):
    # 평가 지표 처리
    import evaluate                                               # 평가 지표 처리 (예: accuracy, recall, precision, f1-score, etc.)

    predict = np.argmax(predictions.predictions, axis=1)  
    accuracy = evaluate.load('accuracy')
    return accuracy.compute(predictions=predict, 
                           references=predictions.label_ids)

# 한국 시간 설정
def get_kst_time():
    kst = pytz.timezone('Asia/Seoul')
    return datetime.now(kst).strftime('%m%d_%H%M')

##### 데이터 로드 함수 정의 #####
def data_load(**context):
    from sklearn.model_selection import train_test_split         # 데이터 분할

    df = pd.read_csv(data_path, sep='\t')
    # df = pd.read_csv(data_path)
    print("컬럼 확인:", df.columns)  # Check the columns
    print("데이터 타입 확인:", df.dtypes)  # Check data types
    print("데이터 상위 몇 행 확인:", df.head())  # Display first few rows
    
    # df = df.sample(n=sample_size, random_state=123)
    df = df.sample(n=min(sample_size, len(df)), random_state=123)
    print(f'데이터 샘플링 완료: {len(df)} rows')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    train_df = Dataset.from_pandas(train_df)
    test_df = Dataset.from_pandas(test_df)
    dataset = ({
        'train': train_df.to_dict(),
        'test': test_df.to_dict()
    })
    print(f'데이터 로드 완료')
    
    ti = context['ti']
    ti.xcom_push(key='dataset', value=dataset)
    ti.xcom_push(key='model_path', value=model_path)

def clean_text_function(examples):
    from konlpy.tag import Okt

    clean_text = []
    morphs_list = []   ##
    nouns_list = []    ##
    pos_list = []      ##
    okt = Okt()
    
    ## 텍스트 정제 ##
    for text in examples['document']:
        if text is None:
            clean_text.append('')
            morphs_list.append('')    ##
            nouns_list.append('')   ##
            pos_list.append('')    ##
        else:
            text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
            text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
            text = re.sub(r'\d+', '', text)  # 숫자 제거
            text = text.lower()  # 소문자로 변환
            text = text.strip()  # 문자열 양쪽 공백 제거
            text = text.replace('br', '')  # 'br' 태그 제거

            morphs = ' '.join(okt.morphs(text))
            nouns = ' '.join(okt.nouns(text))
            pos = ' '.join([f"{word}_{tag}" for word, tag in okt.pos(text)])
            # pos = ' '.join(okt.pos(text))

            clean_text.append(text)
            morphs_list.append(morphs)
            nouns_list.append(nouns)
            pos_list.append(pos)
            
    return {'document': clean_text,
            'morphs': morphs_list,
            'nouns': nouns_list,
            'pos': pos_list}

##### 데이터 전처리 함수 #####
def data_preprocess(**context):
    start_time = time.time()
    ti = context['ti']

    # Pull the dataset from XCom
    dataset_dict = ti.xcom_pull(key='dataset')  # Returns a dictionary

    # Convert the dictionary back into a DatasetDict
    dataset = DatasetDict({
        'train': Dataset.from_dict(dataset_dict['train']),
        'test': Dataset.from_dict(dataset_dict['test'])
    })
    
    dataset_clean = {}
    for split in dataset:
        dataset_clean[split] = dataset[split].map(clean_text_function, batched=True)
    print(f'데이터 정제 완료')
    
    ## 토크나이저 적용 ##
    tokenizer = load_tokenizer()
    dataset_tokenized = {}
    for split in dataset_clean:
        dataset_tokenized[split] = dataset_clean[split].map(
            lambda x: tokenizer(
                x['document'],
                padding='max_length',
                truncation=True,
                max_length=300,
                return_tensors=None
            ),
            batched=True
        )

    ## 라벨 딕셔너리 생성 ##
    label2id = {'1': 1, '0': 0}
    dataset_labeled = {}
    for split in dataset_tokenized:
        dataset_labeled[split] = dataset_tokenized[split].map(
            lambda x: {'label': label2id[str(x['label'])]}
        )
    dataset_labeled = DatasetDict(dataset_labeled)
    # dataset_saved_path = os.path.join(dataset_path, f'processed_dataset_{timestamp}.json')
    # dataset_labeled.save_to_disk(dataset_saved_path)
    print(f'데이터 전처리 완료')
    end_time = time.time()
    preprocess_time = end_time - start_time

    # Push the tokenized dataset back to XCom
    dataset_tokenized_dict = {split: dataset_tokenized[split].to_dict() for split in dataset_tokenized}
    ti.xcom_push(key='dataset_tokenized_dict', value=dataset_tokenized_dict)
    ti.xcom_push(key='label2id', value=label2id)
    ti.xcom_push(key='preprocess_time', value=preprocess_time)


##### 모델 설정 #####
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
    
##### 모델 학습 및 평가 함수 #####
def train_eval_model(**context):
    from transformers import Trainer, TrainingArguments   
    import pickle                               # 텐서 처리. DataLoader: 데이터 로더

    # mlflow 설정
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Kobert_10data")

    # os.makedirs(output_path, exist_ok=True)
    ti = context['ti']
    dataset_tokenized_dict = ti.xcom_pull(key='dataset_tokenized_dict')

    # `DatasetDict`로 변환
    if not isinstance(dataset_tokenized_dict, dict):
        raise ValueError("Expected `dataset_tokenized_dict` to be a dictionary.")

    dataset_labeled = DatasetDict({
        split: Dataset.from_dict(data) for split, data in dataset_tokenized_dict.items()
    })
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8

    ## 라벨 설정 ##
    id2label = {0: '0', 1: '1'}
    label2id = {'0': 0, '1': 1}
    
    ## 모델 설정 ##
    model = CustomBertClassifier(
        model_name,
        num_labels=len(label2id)
    )
    model.to(device)

    # 학습 설정 ##
    args = TrainingArguments(
        output_dir=model_path,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=2e-5,
        eval_strategy='epoch', 
        save_strategy='steps',       
        save_steps=500, 
        save_total_limit=3
    )

    ## Trainer 설정 ##
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_labeled['train'],
        eval_dataset=dataset_labeled['test'],
        tokenizer=load_tokenizer(),
        compute_metrics=compute_accuracy    
    )
    
    # trainer.save_model(model_path)

    ## mlflow 실행 ##
    # mlflow.autolog()
    with mlflow.start_run():
        start_time = time.time()
        ## 모델 학습 ##
        trainer.train()
        end_time = time.time()
        training_time = end_time - start_time
        timestamp = get_kst_time()
        print(f'모델 학습 완료')

        ## 모델 평가 및 예측 ##
        evaluation = trainer.evaluate()
        predictions = trainer.predict(dataset_labeled['test'])
        accuracy_score = compute_accuracy(predictions)
        
        ## 평가 결과 저장 ##
        evaluation_result = {
            "model_name": model_name,
            "eval_loss": evaluation['eval_loss'],                # 평가 손실
            "eval_accuracy": evaluation['eval_accuracy'],        # 평가 정확도
            "predict_accuracy": accuracy_score['accuracy'],      # 예측 정확도
            "eval_runtime": evaluation['eval_runtime'],          # 평가 실행 시간
            "training_time_seconds": training_time               # 학습 시간
        }

        print(f'모델 평가 완료')

        ## mlflow 지표 저장 ##
        mlflow.log_metrics({'test_accuracy': accuracy_score['accuracy'],
                            'eval_loss': evaluation['eval_loss'],
                            'eval_accuracy': evaluation['eval_accuracy'],
                            'eval_runtime': evaluation['eval_runtime']})
        
        ## mlflow 파라미터 저장 ##
        mlflow.log_params({
            'model': model_name,
            'dataset': 'ratings',
            'sample_size': sample_size,
            'epochs': trainer.args.num_train_epochs,
            'batch_size': trainer.args.per_device_train_batch_size,
            'timestamp': timestamp
        })
        # mlflow.set_tags({'dataset': 'ratings',
        #                  'model': 'kobert',
        #                  'timestamp': timestamp})
    
        trainer.save_model(model_path)
        
        ## pkl로 모델 저장 ##
        model_save_pkl = os.path.join(model_path, f'model_{timestamp}.pkl') # pkl로도 저장
        with open(model_save_pkl, 'wb') as f:
            pickle.dump(model, f)
        mlflow_model_name = model_name.replace("/", "_")
        # joblib.dump(model, model_save_pkl) 

        ## mlflow에 모델 저장 ##
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=mlflow_model_name,
            pip_requirements=[
                "torch==2.5.1",
                "cloudpickle==3.1.0"
            ]
        )
        print(f'모델 저장 완료: {model_save_pkl}')


    print(f'모델 학습 및 평가 완료')
    ti.xcom_push(key='evaluation_result', value=evaluation_result)

##### 모델 로드 함수 #####
# def load_saved_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
#     model = CustomBertClassifier(model_name, num_labels=2)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.to(device)
#     # return model
# loaded_model = load_saved_model('path/to/saved/model.pkl')


##### 슬랙 알림 함수 #####
def slack_notification(**context):
    ti = context['ti']
    dataset_dict = ti.xcom_pull(key='dataset')  # XCom에서 딕셔너리 가져오기
    dataset = DatasetDict({
        'train': Dataset.from_dict(dataset_dict['train']),
        'test': Dataset.from_dict(dataset_dict['test'])
    })
    evaluation_result = ti.xcom_pull(key='evaluation_result')
    print(evaluation_result, type(evaluation_result))
    preprocess_time = ti.xcom_pull(key='preprocess_time')
    
    message = f"""
* Dataset 정보 *
- 데이터 경로: {os.path.basename(data_path)}
- 데이터 샘플링 개수: {sample_size}
- 학습 데이터 개수: {len(dataset['train'])}
- 평가 데이터 개수: {len(dataset['test'])}
- 데이터 전처리 시간: {preprocess_time:.2f} seconds

* Model 정보 *
- 모델명: {model_name}
- 저장 경로: {model_path}

* 학습 결과 *
- 평가 손실: {evaluation_result['eval_loss']:.4f}
- 평가 정확도: {evaluation_result['eval_accuracy']:.4f}
- 예측 정확도: {evaluation_result['predict_accuracy']:.4f}
- 학습 시간: {evaluation_result['training_time_seconds']:.2f} seconds

* 퍼포먼스 *
- 평가 실행 시간: {evaluation_result['eval_runtime']:.2f} seconds
- 초당 샘플 개수: {evaluation_result['eval_runtime'] / evaluation_result['predict_accuracy']:.2f}

* MLFlow 정보 *
- 실행 경로: {mlflow.get_artifact_uri()}
    """
    slack_notification = SlackWebhookOperator(
        task_id='send_slack_notification_task',
        slack_webhook_conn_id='slack_webhook',
        message=message,
        username='news_bot',
        dag=context['dag']
    )
    
    # Slack 메시지를 실제로 전송
    slack_notification.execute(context=context)
    print(f'슬랙 알림 완료')

# Airflow 기본 설정
default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 12, 4),
    'retries': 3,
    'env': {
        'NO_PROXY': '*',   # airflow로 외부 요청할 때 
        'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python'   # 파이썬 버전 이슈 해결
    }
}

# DAG 정의
dag = DAG(
    'reviews_ml_training_pipeline_with_mlflow',
    default_args=default_args,
    description='A machine learning pipeline with MLflow logging on review dataset',
    schedule_interval='@daily',
    catchup=False
)

# Task 정의
data_load_task = PythonOperator(
    task_id='data_load',
    python_callable=data_load,
    provide_context=True,
    dag=dag
)

data_preprocess_task = PythonOperator(
    task_id='data_preprocess',
    python_callable=data_preprocess,
    provide_context=True,
    dag=dag
)

train_eval_model_task = PythonOperator(
    task_id='train_eval_model',
    python_callable=train_eval_model,
    provide_context=True,
    dag=dag
)

# Slack 메시지 전송 Task
slack_notification_task = PythonOperator(
    task_id='slack_notification',
    python_callable=slack_notification,
    provide_context=True,
    dag=dag
)

# Task 의존성 설정
data_load_task >> data_preprocess_task >> train_eval_model_task >> slack_notification_task
