# EvalSuite as a small HTTP service: docker build -t evalsuite . && docker run -p 8000:8000 evalsuite
# POST /evaluate {"y_true": [...], "y_pred": [...], "y_prob": [...]}  ·  GET /metrics  ·  GET /health
FROM python:3.12-slim
RUN pip install --no-cache-dir "evalsuite-python[all]" && useradd --create-home evalsuite
USER evalsuite
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["evalsuite", "serve", "--host", "0.0.0.0", "--port", "8000"]
