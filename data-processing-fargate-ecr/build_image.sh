#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="${1:-reddit-processor-fargate:latest}"
shift || true

exec docker build   -f "${SCRIPT_DIR}/Dockerfile"   -t "${IMAGE_TAG}"   "$@"   "${ROOT_DIR}"
