import pandas as pd
import os
import re

import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from safetensors.torch import load_file


# Custom BERT Classifier class (needs to be identical to the training class)
class CustomBertClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return torch.nn.functional.softmax(logits, dim=1)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # HTML 태그 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수문자 제거
    text = re.sub(r'\d+', '', text)  # 숫자 제거
    text = text.lower()  # 소문자로 변환
    text = text.strip()  # 문자열 양쪽 공백 제거
    text = text.replace('br', '')  # 'br' 태그 제거
    return text

def predict_sentiment(text, model_path, model_name='skt/kobert-base-v1'):
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = CustomBertClassifier(model_name, num_labels=2)
    state_dict = load_file(os.path.join(model_path, 'model.safetensors'))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 토크나이저 로드
    tokenizer = KoBERTTokenizer.from_pretrained(
        model_name,
        sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True}
    )
    
    # 텍스트 전처리
    cleaned_text = clean_text(text)
    
    # 토크나이징
    encoded = tokenizer(
        cleaned_text,
        padding='max_length',
        truncation=True,
        max_length=300,
        return_tensors='pt'
    )
    
    # 예측
    with torch.no_grad():
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = outputs.squeeze()
        predicted_class = torch.argmax(probabilities).item()
        
    # 결과 해석
    sentiment = "긍정" if predicted_class == 1 else "부정"
    confidence = probabilities[predicted_class].item()
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': f'{confidence:.2%}',
        'probabilities': {
            '부정': f'{probabilities[0].item():.2%}',
            '긍정': f'{probabilities[1].item():.2%}'
        }
    }

# 사용 예시
# if __name__ == "__main__":
#     # 모델 경로 설정
#     model_path = 'kobert_konply'  # 실제 모델 경로로 변경하세요
    
#     # CSV 파일 읽기
#     try:
#         # CSV 파일 읽기
#         df = pd.read_csv('data/cgv_reviews.csv', encoding='utf-8')
#         print(f"총 {len(df)} 개의 리뷰를 읽었습니다.")
        
#         # 리뷰 텍스트가 있는 컬럼 이름을 'review'로 가정
#         # 실제 컬럼 이름에 맞게 수정하세요
#         reviews = df['review'].tolist()
        
#         # 각 리뷰에 대해 감성 분석 수행
#         results = []
#         for i, review in enumerate(reviews, 1):
#             if pd.isna(review):  # 빈 리뷰 건너뛰기
#                 continue
                
#             result = predict_sentiment(str(review), model_path)
#             results.append(result)
            
#             # 진행 상황 출력 (100개마다)
#             if i % 100 == 0:
#                 print(f"{i}개의 리뷰 처리 완료...")
            
#         # 결과를 데이터프레임으로 변환
#         results_df = pd.DataFrame(results)
        
#         # 결과를 CSV 파일로 저장
#         output_path = 'data/cgv_reviews_with_sentiment.csv'
#         results_df.to_csv(output_path, index=False, encoding='utf-8')
#         print(f"\n분석 완료! 결과가 {output_path}에 저장되었습니다.")
        
#         # 간단한 통계 출력
#         total = len(results_df)
#         positive = (results_df['sentiment'] == '긍정').sum()
#         negative = (results_df['sentiment'] == '부정').sum()
        
#         print(f"\n=== 분석 통계 ===")
#         print(f"전체 리뷰 수: {total}")
#         print(f"긍정 리뷰: {positive} ({positive/total*100:.1f}%)")
#         print(f"부정 리뷰: {negative} ({negative/total*100:.1f}%)")
        
#     except FileNotFoundError:
#         print("Error: 'data/cgv_reviews.csv' 파일을 찾을 수 없습니다.")
#     except Exception as e:
#         print(f"에러 발생: {str(e)}") 