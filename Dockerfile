FROM python:3.11-slim

# ---- System + dev tools (VS Code friendly) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    openssh-client \
    awscli \
    unzip \
    zip \
    nano \
    vim \
    less \
    procps \
    lsof \
    net-tools \
    iputils-ping \
    ripgrep \
    jq \
    # runtime libs often needed by numpy/pyarrow/torch on slim:
    libgomp1 \
    # (optional) build tools; keep if you expect any source builds:
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Create a non-root user for VS Code ----
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && mkdir -p /workspace \
    && chown -R ${USERNAME}:${USERNAME} /workspace

WORKDIR /workspace

# ---- Python tooling ----
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python deps (as root is fine; they go into the image)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Caches (HF, pip) placed somewhere writable by the non-root user ----
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/${USERNAME}/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/${USERNAME}/.cache/huggingface/transformers \
    HF_DATASETS_CACHE=/home/${USERNAME}/.cache/huggingface/datasets

RUN mkdir -p /home/${USERNAME}/.cache \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache

# Switch to non-root for VS Code
USER ${USERNAME}

# Default shell
CMD ["bash"]
