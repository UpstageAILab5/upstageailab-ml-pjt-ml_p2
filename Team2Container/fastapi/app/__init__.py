# app/__init__.py

# 패키지 버전 정의 (선택 사항)
__version__ = "1.0.0"

# FastAPI 인스턴스를 다른 모듈에서 사용할 수 있도록 가져오기
from .app import app

# __all__을 사용하여 외부에 노출할 항목 정의
__all__ = ["app"]