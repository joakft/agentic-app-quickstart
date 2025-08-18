# Use the same base image as in devcontainer.json
FROM python:3.13-slim

# Install system utilities (like your "common-utils" + git features)
RUN apt-get update && apt-get install -y \
    git \
    zsh \
    curl \
    sudo \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Create vscode-like user (optional, matches your devcontainer.json)
RUN useradd -m vscode && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER vscode
WORKDIR /workspace

# Copy your code
COPY . /workspace

# Run your setup script (if it installs Python deps, tools, etc.)
RUN bash .devcontainer/setup.sh || true

# Default: just drop into shell
CMD ["/bin/bash"]
