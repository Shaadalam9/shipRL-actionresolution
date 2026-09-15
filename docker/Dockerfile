# Training image for the ASV path-following experiments.
# Build with docker/build.sh, run with docker/run.sh.

FROM tensorflow/tensorflow:2.9.1-gpu

LABEL maintainer="Marine Autonomous Vessels Laboratory, IIT Madras"
LABEL version="3.0"
LABEL description="Reinforcement learning controllers for ASV path following"

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        libsm6 libxext6 libxrender1 libfontconfig1 libgl1 \
        sudo xauth \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements-tf291.txt /requirements.txt
RUN pip install -r /requirements.txt

ARG USERNAME=docker
RUN useradd -m $USERNAME && \
        echo "$USERNAME:$USERNAME" | chpasswd && \
        usermod --shell /bin/bash $USERNAME && \
        usermod -aG sudo $USERNAME && \
        echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME

# The repository is bind-mounted here by run.sh, so edits on the host take
# effect in the container immediately.
ENV PYTHONPATH=/home/docker/compare_rl
WORKDIR /home/docker/compare_rl
USER $USERNAME

CMD ["bash"]
