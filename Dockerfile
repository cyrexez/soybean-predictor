# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_test.py predict.py app.py ./

RUN python train_test.py

# Run predictions first, then start API
CMD ["sh", "-c", "python predict.py && python app.py"]