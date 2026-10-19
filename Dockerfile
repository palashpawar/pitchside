FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-azure.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-azure.txt
COPY pitchside ./pitchside
COPY web ./web
COPY data/model_report.json ./data/model_report.json
ENV PITCHSIDE_SEED=4 PORT=8000
# Pre-simulate the demo match so the first viewer does not wait for it.
RUN python -c "from pitchside.server import get_match; get_match()"
EXPOSE 8000
CMD ["sh", "-c", "uvicorn pitchside.server:app --host 0.0.0.0 --port ${PORT}"]
