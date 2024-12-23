FROM python:3.12-slim

WORKDIR /app/src/ai

COPY ./requirements.txt .

RUN apt-get update && apt-get install dumb-init

RUN pip install --upgrade pip

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers

RUN pip install --no-cache-dir -r requirements.txt

COPY ./srcs/ /app/src/ai/
COPY ./tools/ /usr/bin/local

RUN chmod +x /usr/bin/local/script.sh

EXPOSE $AI_PORT

ENTRYPOINT [ "/usr/bin/dumb-init", "--", "sh", "/usr/bin/local/script.sh" ]