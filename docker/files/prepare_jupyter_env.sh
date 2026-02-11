#!/bin/bash

cd /workspace
openssl enc -d -aes-256-cbc -out /workspace/.aws/credentials -in /workspace/.aws/credentials.enc -K $mykey -iv $iv
python3 -m venv /workspace/venv && /workspace/venv/bin/pip install --upgrade pip && /workspace/venv/bin/pip install wandb "trl>=0.20.0" "peft>=0.17.0" "transformers>=5.0.0" nbformat nbclient huggingface huggingface_hub
/workspace/venv/bin/pip install -U "nvidia-modelopt[all]"
git clone https://github.com/AlexandreFenyo/ovh-container-gpt-oss
cp ovh-container-gpt-oss/docker/files/ft.py .
cp ovh-container-gpt-oss/docker/files/merge.py .
cp ovh-container-gpt-oss/docker/files/awscli-exe-linux-x86_64.zip awscliv2.zip
unzip awscliv2.zip
./aws/install --bin-dir /workspace/data/bin --install-dir /workspace/data/aws-cli --update
/workspace/data/bin/aws --version
export PATH=/workspace/data/bin:$PATH
aws s3 cp s3://cnam-models/gpt-oss-20b gpt-oss-20b --recursive

/workspace/venv/bin/python ft.py
