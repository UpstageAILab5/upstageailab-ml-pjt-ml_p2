# 공통적으로 필요한 것 정의

# base.Dockerfile
FROM python:3.10-slim

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    git \
    openjdk-17-jdk \
    firefox-esr \
    wget \
    libnss3 \
    fonts-liberation \
    libgconf-2-4 \
    && wget https://github.com/mozilla/geckodriver/releases/download/v0.32.2/geckodriver-v0.32.2-linux-aarch64.tar.gz \
    && tar -xvzf geckodriver-v0.32.2-linux-aarch64.tar.gz \
    && mv geckodriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/geckodriver \
    && rm geckodriver-v0.32.2-linux-aarch64.tar.gz \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# requirements.txt 파일 복사 및 패키지 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create the matching directory and link it to the actual JDK path
RUN mkdir -p /opt/homebrew/opt/openjdk@17 \
    && ln -s /usr/lib/jvm/java-17-openjdk-amd64 /opt/homebrew/opt/openjdk@17

ENV JAVA_HOME=/opt/homebrew/opt/openjdk@17
ENV PATH=$JAVA_HOME/bin:$PATH