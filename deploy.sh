#!/usr/bin/env bash
set -euo pipefail

APP_NAME="digital-twin-scheduler"
IMAGE_NAME="${APP_NAME}:latest"
CONTAINER_NAME="${APP_NAME}"
ENV_FILE="/opt/digital-twin-scheduler/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}"
  exit 1
fi

docker build -t "${IMAGE_NAME}" .

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  -p 5000:5000 \
  --env-file "${ENV_FILE}" \
  "${IMAGE_NAME}"

docker ps --filter "name=${CONTAINER_NAME}"
