FROM python:3.10-slim

# ----------------------------
# SET WORKDIR
# ----------------------------
WORKDIR /app

# ----------------------------
# INSTALL SYSTEM DEPENDENCIES
# ----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# COPY REQUIREMENTS
# ----------------------------
COPY requirements.txt .

# ----------------------------
# INSTALL PYTHON DEPENDENCIES
# ----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# COPY PROJECT FILES
# ----------------------------
COPY . .

# ----------------------------
# EXPOSE PORT (HF expects 7860)
# ----------------------------
EXPOSE 7860

# ----------------------------
# START BOTH SERVICES
# ----------------------------
CMD bash -c "\
echo 'Starting FastAPI...' && \
uvicorn api.app:app --host 0.0.0.0 --port 8000 --log-level debug \
"