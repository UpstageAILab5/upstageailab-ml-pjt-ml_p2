import boto3
from botocore.exceptions import ClientError
import logging
import os
from io import BytesIO
from PIL import Image

class s3_client:
    def __init__(self, access_key, secret_access_key):
        """
        :param access_key: S3 액세스 키
        :param secret_access_key: S3 비밀 액세스 키
        """
        self.BUCKET = 'fastcampus-ml-p2-bucket'
        self.AWS_DEFAULT_REGION = "ap-northeast-2"
        self.s3 = boto3.client('s3', 
                               aws_access_key_id=access_key, 
                               aws_secret_access_key=secret_access_key, 
                               region_name=self.AWS_DEFAULT_REGION)

    def upload_file(self, file_name:str, bucket_path:str, object_name:str=None)->bool:
        """ S3 버킷에 파일을 업로드합니다.
        
        :param file_name: 업로드할 파일
        :param bucket_path: 업로드될 버킷의 경로
        :param object_name: S3 객체이름. 없으면 file_name 사용
        :return: 파일이 업로드되면 True, 아니면 False
        """
        
        # S3 객체이름이 정의되지 않으면, file_name을 사용
        if object_name is None:
            object_name = os.path.basename(file_name)
        
        # 파일 업로드
        try:
            resposne = self.s3.upload_file(file_name, self.BUCKET, bucket_path + object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True
    
    def get_images(self)->list:

        # S3 버킷 이름과 파일들이 있는 경로
        prefix = 'image/'

        # S3에서 파일 목록을 가져오기
        response = self.s3.list_objects_v2(Bucket=self.BUCKET, Prefix=prefix)

        # 파일 목록이 있는 경우에만 실행
        if 'Contents' in response:
            # 이미지 파일을 저장할 리스트
            image_list = []
            
            for obj in response['Contents']:
                file_key = obj['Key']
                
                # 이미지 파일을 S3에서 읽기
                file_response = self.s3.get_object(Bucket=self.BUCKET, Key=file_key)
                file_content = file_response['Body'].read()
                
                # 이미지 파일을 메모리에서 바로 열기 전에 일부 바이트를 확인
                print(file_content[:10])  # 첫 10바이트를 출력하여 이미지 형식 점검

                try:
                    image = Image.open(BytesIO(file_content))
                    
                    # 이미지 객체를 리스트에 추가
                    image_list.append(image)
                except IOError:
                    print(f"이미지 파일 '{file_key}'을(를) 열 수 없습니다.")
            
            return image_list
        else:
            print("파일이 존재하지 않습니다.")