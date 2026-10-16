FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel

WORKDIR /workspace

COPY . .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

CMD ["python", "main.py"]
