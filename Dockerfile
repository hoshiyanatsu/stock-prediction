FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (curl は HEALTHCHECK 用)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 本番ビルド時のデフォルト資材（開発時は compose のボリュームで上書きされます）
COPY app.py .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.headless", "true", "--server.address", "0.0.0.0"]
