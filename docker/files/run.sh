#!/bin/bash

# variables d'env attendues :
# mykey : clé de chiffrement AES-256 des fichiers .enc
# iv : IV utilisé avec le chiffrement AES-256
# wandbkey : clé d'API wandb
# hfkey : clé d'API Hugging Face

echo PREPARING FILESYSTEM
openssl enc -d -aes-256-cbc -out /workspace/.aws/credentials -in /workspace/.aws/credentials.enc -K $mykey -iv $iv
git clone https://github.com/AlexandreFenyo/ovh-container-gpt-oss
cp ovh-container-gpt-oss/docker/files/ft.py .
cp ovh-container-gpt-oss/docker/files/merge.py .
cp ovh-container-gpt-oss/docker/files/query.py .
cp ovh-container-gpt-oss/docker/files/run-from-git.sh .
echo DONE FILESYSTEM

bash ./run-from-git.sh
