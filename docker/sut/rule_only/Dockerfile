FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY isynkgr /app/isynkgr
COPY benchmark /app/benchmark
RUN pip install --no-cache-dir . requests==2.32.3 pydantic==2.9.2 matplotlib==3.9.2
ENV PYTHONUNBUFFERED=1
CMD ["python","-u","-m","benchmark.run_sut"]
