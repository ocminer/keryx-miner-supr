# ocminersupr/keryx-miner-supr — env-driven keryx PoM v4 miner image.
#
# The modern release binary bundles its CUDA runtime libs (in ./lib), so the slim -base CUDA image
# is enough — the NVIDIA container runtime injects the driver's libcuda at run time. Build from the
# repo root AFTER a modern build:  docker build -f docker/Dockerfile -t ocminersupr/keryx-miner-supr .
# (the build context must contain hiveos/dist-modern/ from `hiveos/build-offline.sh … dist-modern modern`).
FROM nvidia/cuda:12.9.1-base-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget htop tmux nano openssh-server \
        libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

WORKDIR /app
# The full modern dist: binary + plugins (libkeryxcuda.so/libkeryxopencl.so) + bundled CUDA libs (lib/)
# + the in-process llama.cpp engine (libkeryx-llama.so) for OPoI inference. Built by build-offline.sh.
COPY hiveos/dist-modern/keryx-miner-supr /app/keryx-miner-supr
COPY hiveos/dist-modern/libkeryxcuda.so /app/libkeryxcuda.so
COPY hiveos/dist-modern/libkeryxopencl.so /app/libkeryxopencl.so
COPY hiveos/dist-modern/lib /app/lib
COPY docker/entrypoint.sh /app/entrypoint.sh
# libkeryx-llama.so is staged in during packaging; copy it if present (build ignores a missing one
# by using a wildcard that always matches at least entrypoint.sh).
COPY hiveos/dist-modern/libkeryx-llama.s[o] docker/entrypoint.sh /app/
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
