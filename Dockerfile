FROM python:3.13

WORKDIR /src

RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 && \
  python3 -m pip install transformers==4.49.0 wyoming==1.6.0

# TODO: Remove ffmpeg dependency and pass an np.array to the pipeline instead
RUN apt-get update && \
  apt-get install --no-install-recommends --yes ffmpeg && \
  rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 10300/tcp
VOLUME /data
LABEL org.opencontainers.image.source=https://github.com/AlexGustafsson/wyoming-kb-whisper

ENTRYPOINT ["python3", "-m", "wyoming_kb_whisper", "--uri", "tcp://0.0.0.0:10300", "--data-dir", "/data"]
