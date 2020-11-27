FROM python:3.7-slim
RUN apt-get -yqq update && apt-get install -yqq g++ libopenblas-base libopenblas-dev swig
ENV G2P_MODEL_DIR=/app/fairseq_g2p/
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
EXPOSE 8000
VOLUME /app/final.mdl
COPY . /app
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-"]
CMD ["app:app"]
