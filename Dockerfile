ARG OPENVIKING_IMAGE
FROM ${OPENVIKING_IMAGE}

ARG PYTHON_BIN=/app/.venv/bin/python
ARG PIP_BIN=/usr/local/bin/pip

USER root
WORKDIR /tmp/openviking-seekdb
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN "${PIP_BIN}" --python "${PYTHON_BIN}" install --no-cache-dir . \
    && rm -rf /tmp/openviking-seekdb

WORKDIR /app/.openviking
