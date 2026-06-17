# ============================================================
# Dockerfile.api — FastAPI 后端镜像
# ============================================================

FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/textbooks /app/data/courseware /app/data/exam_banks

EXPOSE 8000

CMD ["python", "-m", "app.edu_api"]
