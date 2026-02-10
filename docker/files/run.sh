#!/bin/bash

# variables d'env attendues :
# mykey : clé de chiffrement AES-256 des fichiers .enc
# iv : IV utilisé avec le chiffrement AES-256
# wandbkey : clé d'API wandb
# hfkey : clé d'API Hugging Face

openssl enc -d -aes-256-cbc -out /workspace/.aws/credentials -in /workspace/.aws/credentials.enc -K $mykey -iv $iv
git clone https://github.com/AlexandreFenyo/ovh-container-gpt-oss
cp ovh-container-gpt-oss/docker/files/ft.py .
cp ovh-container-gpt-oss/docker/files/merge.py .
aws s3 cp s3://cnam-models/gpt-oss-20b gpt-oss-20b --recursive

/workspace/venv/bin/python3 ft.py
/workspace/venv/bin/python3 merge.py
cp gpt-oss-20b-tokenizer/* gpt-oss-20b-merged
( cd TensorRT-Model-Optimizer/examples/gpt-oss ; /workspace/venv/bin/python3 convert_oai_mxfp4_weight_only.py --model_path /workspace/gpt-oss-20b-merged --output_path /workspace/gpt-oss-20b-merged-mxfp4 )
aws s3 rm s3://cnam-models/gpt-oss-20b-merged-mxfp4 --recursive
aws s3 cp gpt-oss-20b-merged-mxfp4 s3://cnam-models/gpt-oss-20b-merged-mxfp4 --recursive
