FROM python:3.11-slim

WORKDIR /app

# gcc needed for some scipy/numpy wheel builds on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install runtime + API dependencies from explicit list.
# This layer is cached as long as these versions don't change,
# regardless of source code modifications.
RUN pip install --no-cache-dir \
    "numpy>=1.26" \
    "scipy>=1.11" \
    "scikit-learn>=1.3" \
    "pydantic>=2.5" \
    "rich>=13.7" \
    "typer>=0.9" \
    "jinja2>=3.1" \
    "fastapi>=0.109" \
    "uvicorn[standard]>=0.27"

# Copy source and install the package itself
COPY pyproject.toml README.md ./
COPY evalkit/ ./evalkit/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "evalkit.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
