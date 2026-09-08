# ocminersupr/keryx-miner-supr — env-driven keryx PoM v4 miner image.
#
# The modern release binary bundles its CUDA runtime libs (in ./lib), so the slim -base CUDA image
# is enough — the NVIDIA container runtime injects the driver's libcuda at run time. Build from the
# repo root AFTER a modern build:  docker build -f docker/Dockerfile -t ocminersupr/keryx-miner-supr .
# (the build context must contain hiveos/dist-modern/ from `hiveos/build-offline.sh … dist-modern modern`).
FROM nvidia/cuda:12.9.1-base-ubuntu24.04

# openssh-server generates host keys during installation. Remove them in this same layer:
# public image layers must never contain shared private host keys. Each container creates
# its own keys on first start (see entrypoint.sh).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget htop tmux nano openssh-server \
        libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && rm -f /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_rsa_key.pub \
        /etc/ssh/ssh_host_ecdsa_key /etc/ssh/ssh_host_ecdsa_key.pub \
        /etc/ssh/ssh_host_ed25519_key /etc/ssh/ssh_host_ed25519_key.pub

WORKDIR /app
# NVIDIA-only modern PoM host, matching CUDA plugin and runtime closure. AMD cards need the
# separate pom-opencl package; loading an OpenCL plugin into this host cannot change its backend.
COPY hiveos/dist-modern/keryx-miner-supr /app/keryx-miner-supr
COPY hiveos/dist-modern/libkeryxcuda.so /app/libkeryxcuda.so
COPY hiveos/dist-modern/lib /app/lib
COPY docker/entrypoint.sh /app/entrypoint.sh
# Both GPU inference engines are mandatory, just as in the Linux archives. The baseline-CPU
# variant still uses CUDA for inference; omitting it breaks rigs whose CPUs cannot execute AVX2.
COPY hiveos/dist-modern/libkeryx-llama.so /app/libkeryx-llama.so
COPY hiveos/dist-modern/libkeryx-llama-noavx.so /app/libkeryx-llama-noavx.so
RUN chmod +x /app/keryx-miner-supr /app/entrypoint.sh

# Env contract — see docker/entrypoint.sh. Defaults mine keryx on suprnova, --tier auto, /models.
ENV KERYX_WALLET= \
    KERYX_WORKER= \
    KERYX_POOL=stratum+tcp://krx.suprnova.cc:4404 \
    KERYX_BACKUP=stratum+tcp://krx.suprnova.cc:4401 \
    KERYX_TIER=auto \
    KERYX_MODEL_DIR=/models \
    KERYX_DEVICES= \
    KERYX_INFERENCE_GPU= \
    KERYX_EXTRA_ARGS= \
    ROOT_PASSWORD=keryx \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

VOLUME ["/models"]
EXPOSE 22
CMD ["/app/entrypoint.sh"]
