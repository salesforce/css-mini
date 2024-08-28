FROM --platform=linux/amd64 python:3.10-slim as builder

WORKDIR /opt/

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential libffi-dev libc6-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY . ./css

RUN pip install --no-cache-dir './css[deploy]'

FROM --platform=linux/amd64 python:3.10-slim

LABEL maintainer="Salesforce MLE"

# Copy only the installed packages and binaries from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Define serving entrypoint
ENTRYPOINT ["serve"]
