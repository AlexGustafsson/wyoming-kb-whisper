# Wyoming KB Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for
[KBLab's Swedish whisper models](https://huggingface.co/collections/KBLab/kb-whisper-67af9eafb24da903b63cc4aa)

This is essentially a fork of <https://github.com/rhasspy/wyoming-faster-whisper>
with the sole purpose of supporting KBLab's Swedish whisper model.

No GPU support for now as the resulting image is huge and CPU seems to be as
fast for the small models on modern systems.

**NOTE**: Models are downloaded temporarily to the `HF_HUB_CACHE` directory,
which defaults to `~/.cache/huggingface/hub`. You may need to adjust this
environment variable when using a read-only root filesystem (e.g.,
`HF_HUB_CACHE=/tmp`).

## Running using Docker

```shell
docker run --rm -it \
  -p 10300:10300 \
  -v "$PWD/data:/data" \
  ghcr.io/alexgustafsson/wyoming-kb-whisper \
  --model base
```

See full example for Kubernetes in `deployment.yaml`.
