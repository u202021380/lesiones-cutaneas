# Imagen para Hugging Face Spaces (SDK Docker) o Render
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
RUN useradd -m -u 1000 usuario
WORKDIR /home/usuario/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=usuario app ./app
COPY --chown=usuario models ./models
USER usuario
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
