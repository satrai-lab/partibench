# ---------- Base Image ----------
FROM python:3.9-slim

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working Directory ----------
WORKDIR /app

# ---------- Python Dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Project ----------
COPY . .

# ---------- Expose Deployment Port ----------
EXPOSE 6999

# ---------- Default Command ----------
CMD ["python", "local_deploy.py"]
