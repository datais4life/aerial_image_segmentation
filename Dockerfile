FROM python:3.9.16-slim-buster
WORKDIR /inference
COPY . /inference
EXPOSE 8505
RUN apt update -y\
    && pip install -U pip \
    && pip install -r requirements.txt
ENTRYPOINT [ "streamlit", "run", "inference.py" ]