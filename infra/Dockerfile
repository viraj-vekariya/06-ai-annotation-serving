# Serving image, built to fit a 512 MB instance.
#
# THE MEASUREMENT THAT SHAPED THIS FILE. The obvious image - torch, transformers, MiniLM,
# the index - was measured at 456 MB resident after one real query. That fits 512 MB with
# ~56 MB spare, which is not enough to survive a second concurrent request. The breakdown
# said where it went:
#
#     baseline python   12 MB      + transformers  412 MB   <- 215 MB is the LIBRARY
#     + numpy           32 MB      + MiniLM        417 MB
#     + torch          197 MB      + one query     456 MB
#
# `transformers` costs more than the model it loads. It is needed to BUILD the artifacts -
# load the checkpoint, trace it, export the tokenizer - and not at all to run them. So the
# build stage carries it and the runtime stage does not: the encoder ships as a TorchScript
# graph and the tokenizer as the standalone `tokenizers` library. Re-measured: 334 MB, with
# 178 MB of headroom.
#
# CPU-only torch: the default wheel bundles CUDA at ~2.5 GB against ~200 MB here, and
# nothing in the serving path will ever see a GPU.
#
# THE LANGUAGE MODEL IS NOT INSTALLED. The ablation measured it at 0.6040 against the
# retriever's 0.8920, so it is not on the serving path - and shipping a 250M-parameter
# model the service never calls would be carrying the conclusion's opposite.

FROM python:3.12-slim AS build
WORKDIR /build
ENV PYTHONUNBUFFERED=1 HF_HOME=/opt/hf

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir transformers tokenizers numpy

COPY data/ ./data/
COPY src/ ./src/
COPY serve/ ./serve/
COPY tools/ ./tools/
COPY outputs/results.json ./outputs/
COPY artifacts/serving.json ./artifacts/

# 1. Trace MiniLM to TorchScript. Verifies the traced graph against the original before
#    writing - an encoder that silently drifted would change every prediction while every
#    test still passed. Measured difference: 0.00e+00.
RUN python tools/export_encoder.py

# 2. Rebuild the index. `index.npz` is 14 MB of derived floats and is deliberately not in
#    git, so it has to be regenerated - and regenerating it is exactly where a wrong
#    encoder or a shuffled corpus would slip through unnoticed. The script replays the
#    headline measurement (0.8920 top-1 on the first 1,500 test messages) and fails the
#    BUILD if it does not reproduce it, so the image can only ship the system that was
#    actually evaluated.
RUN python tools/build_index.py

# ---------------------------------------------------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 annotate

# No transformers, and no Hugging Face at all: the runtime never reaches out to a third
# party, so a cold start cannot be slowed or failed by someone else's outage.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir tokenizers numpy fastapi "uvicorn[standard]" pydantic && \
    rm -rf /root/.cache

WORKDIR /app
COPY serve/ ./serve/
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY outputs/results.json ./outputs/
COPY --from=build /build/artifacts/serving.json \
                  /build/artifacts/index.npz \
                  /build/artifacts/encoder_traced.pt \
                  /build/artifacts/tokenizer.json \
                  /build/artifacts/encoder_meta.json ./artifacts/

RUN chown -R annotate:annotate /app
USER annotate

ENV PYTHONUNBUFFERED=1 \
    ANNOTATION_ARTIFACTS=/app/artifacts \
    OMP_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1

EXPOSE 8600
# Readiness, not liveness: this must fail while the index loads and warms so the platform
# holds traffic back rather than killing the container.
HEALTHCHECK --interval=15s --timeout=4s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8600/health/ready',timeout=3).status==200 else 1)"

# ${PORT} because the platform assigns it; 8600 locally where it does not.
CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT:-8600} --workers ${WEB_CONCURRENCY:-1}"]
