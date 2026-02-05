FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
LABEL maintainer="alex@fenyo.net"
ENV TERM=vt100
ENV LANG=C.UTF-8
WORKDIR /workspace
ENV HOME=/workspace
# RUN yes | unminimize
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends software-properties-common ca-certificates libxml2 build-essential curl wget iproute2 net-tools less zsh vim inetutils-telnet inetutils-ping jq htop nvidia-utils-590-server unzip python3.12-venv pip git
COPY files/awscli-exe-linux-x86_64.zip .
RUN unzip awscli-exe-linux-x86_64.zip
COPY files/TensorRT-Model-Optimizer.tar.gz .
RUN tar zxf TensorRT-Model-Optimizer.tar.gz
COPY files/gpt-oss-20b-tokenizer.tar.gz .
RUN tar zxf gpt-oss-20b-tokenizer.tar.gz
RUN ./aws/install --update
RUN mkdir /workspace/.aws
COPY files/aws/config /workspace/.aws/config
COPY files/aws/credentials.enc /workspace/.aws/credentials.enc
RUN git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer
RUN python3 -m venv /workspace/venv && /workspace/venv/bin/pip install --upgrade pip && /workspace/venv/bin/pip install wandb "trl>=0.20.0" "peft>=0.17.0" "transformers>=5.0.0" nbformat nbclient "nvidia-modelopt[all]" huggingface huggingface_hub
RUN sed -i s/:1000:1000:/:42420:42420:/ /etc/passwd
RUN sed -i s/:1000:/:42420:/ /etc/group
RUN chown -R 42420:42420 /workspace
COPY files/ft.py.enc .
COPY files/run.sh .
RUN chown -R 42420:42420 ft.py.enc run.sh
CMD ["bash", "/workspace/run.sh"]
