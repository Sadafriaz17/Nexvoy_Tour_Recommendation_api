FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as a normal user, not root (also what Hugging Face expects).
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# requirements.txt pins scikit-learn==1.6.1, the exact version that saved nexvoy_model.pkl
# (see "sklearn_version" in nexvoy_model_info.json). app.py refuses to start if they differ.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user:user . .

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
