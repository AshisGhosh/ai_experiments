ARG BASE_IMAGE=nvidia/cuda
ARG BASE_TAG=11.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}:${BASE_TAG}

RUN apt-get update && apt-get install -y \
    sudo \
    git \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    python-is-python3 \
    curl \
    bash-completion \
    tmux
    
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv pip install --upgrade pip --system

COPY requirements.txt .
RUN uv pip install -r requirements.txt --system

ARG CONTAINER_USER=${USER:-hostuser}
ARG CONTAINER_UID=1000
ARG CONTAINER_GID=1000
RUN if getent passwd ${CONTAINER_UID} >/dev/null; then \
        # get existing username with this uid
        EXISTING_USER=$(getent passwd ${CONTAINER_UID} | cut -d: -f1); \
        # if username doesn't match container user, update username
        if [ "$EXISTING_USER" != "${CONTAINER_USER}" ]; then \
        usermod -l ${CONTAINER_USER} $EXISTING_USER; \
        groupmod -n ${CONTAINER_USER} $EXISTING_USER 2>/dev/null || true; \
        mkdir -p /home/${CONTAINER_USER} 2>/dev/null || true; \
        chown ${CONTAINER_UID}:${CONTAINER_GID} /home/${CONTAINER_USER}; \
        fi; \
    else \
        # create new user if uid doesn't exist
        groupadd --gid ${CONTAINER_GID} ${CONTAINER_USER} 2>/dev/null || true; \
        useradd -m -u ${CONTAINER_UID} -g ${CONTAINER_GID} -s /bin/bash ${CONTAINER_USER}; \
    fi && \
    # Add user to sudo group and set password
    usermod -aG sudo ${CONTAINER_USER} && \
    echo "${CONTAINER_USER}:${CONTAINER_USER}" | chpasswd && \
    # Configure sudoers
    echo "${CONTAINER_USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chmod 0440 /etc/sudoers

USER ${CONTAINER_USER}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/${CONTAINER_USER}/.cargo/bin:${PATH}"
RUN cargo install just

ENTRYPOINT ["generative_models/docker/entrypoint.sh"]

CMD ["/bin/bash"]