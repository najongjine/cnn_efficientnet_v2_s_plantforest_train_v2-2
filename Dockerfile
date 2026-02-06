# 1. Python 3.11 사용
FROM python:3.11-slim

WORKDIR /app

# 2. 필수 패키지 설치 (수정됨)
# llama-cpp-python 빌드를 위해 build-essential(gcc 포함)과 cmake가 필요합니다.
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 3. 유저 설정
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
ENV HF_HOME=/app/.cache/huggingface

COPY --chown=user . .

# 4. 의존성 설치 (권장 수정)
# requirements.txt에서 llama-cpp-python을 제거하고 따로 설치하는 것이 캐싱 및 디버깅에 유리합니다.
# 하지만 기존 방식을 유지한다면 아래와 같이 컴파일러가 있으면 빌드 에러가 사라집니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]