FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "media_intel.api.main:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
