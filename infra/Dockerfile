# Serving image.
#
# CPU-only torch: the default wheel bundles CUDA at ~2.5GB against ~200MB here, and
# nothing in the serving path will ever see a GPU - MiniLM runs a 22.7M-parameter forward
# pass in about 5ms on CPU.
#
# The LANGUAGE MODEL IS NOT INSTALLED. The ablation measured it at 0.6040 against the
# retriever's 0.8920, so it is not on the serving path - and shipping a 250M-parameter
# model the service never calls would be carrying the conclusion's opposite.
FROM python:3.12-slim AS build
WORKDIR /build
RUN pip install --no-cache-dir --prefix=/install \
        --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir --prefix=/install \
        transformers numpy fastapi "uvicorn[standard]" pydantic

# Pre-download MiniLM into the image. Without this the first container start reaches out
# to Hugging Face, which makes a third party a hard startup dependency and adds ~20s to
# every cold start.
ENV HF_HOME=/opt/hf
RUN PYTHONPATH=/install/lib/python3.12/site-packages python -c "\
from transformers import AutoModel, AutoTokenizer; \
m='sentence-transformers/all-MiniLM-L6-v2'; \
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

# ---------------------------------------------------------------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 annotate
COPY --from=build /install /usr/local
COPY --from=build /opt/hf /opt/hf
WORKDIR /app

COPY serve/ ./serve/
COPY src/ ./src/
COPY dashboard/ ./dashboard/
# The index and the derived threshold, produced by the pipeline. Baking them in means the
# service serves exactly what was measured.
COPY artifacts/index.npz artifacts/serving.json ./artifacts/
COPY outputs/results.json ./outputs/

RUN chown -R annotate:annotate /app && chmod -R a+rX /opt/hf
USER annotate

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    ANNOTATION_ARTIFACTS=/app/artifacts

EXPOSE 8600
# Readiness, not liveness: this must fail while the index loads and warms so the platform
# holds traffic back rather than killing the container.
HEALTHCHECK --interval=15s --timeout=4s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8600/health/ready',timeout=3).status==200 else 1)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8600", "--workers", "1"]
