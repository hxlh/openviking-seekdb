ARG OPENVIKING_IMAGE
FROM ${OPENVIKING_IMAGE}

ARG PYTHON_BIN=python3
USER root
COPY . /tmp/openviking-seekdb
RUN "${PYTHON_BIN}" -m pip install --no-cache-dir /tmp/openviking-seekdb \
    && rm -rf /tmp/openviking-seekdb
