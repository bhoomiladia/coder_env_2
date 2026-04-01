FROM python:3.10-slim
WORKDIR /app
COPY . .

# Install in editable mode so it links 'server.app:main'
RUN pip install --no-cache-dir -e .

EXPOSE 7860

# Run the script defined in pyproject.toml
CMD ["server"]