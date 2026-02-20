#!/bin/bash

# variables d'env attendues :
# mykey : clé de chiffrement AES-256 des fichiers .enc
# iv : IV utilisé avec le chiffrement AES-256
# wandbkey : clé d'API wandb
# hfkey : clé d'API Hugging Face
# dataset_name : ex. : fenyo/FAQ-MES
# wandb_project : ex. : fenyo-FAQ-MES
# wandb_run : ex. : gpt-oss-20b-FAQ-MES


openssl enc -d -aes-256-cbc -out /workspace/.aws/credentials -in /workspace/.aws/credentials.enc -K $mykey -iv $iv
git clone https://github.com/AlexandreFenyo/ovh-container-gpt-oss
cp ovh-container-gpt-oss/docker/files/ft.py .
cp ovh-container-gpt-oss/docker/files/merge.py .
cp ovh-container-gpt-oss/docker/files/query.py .
aws s3 cp s3://cnam-models/gpt-oss-20b gpt-oss-20b --recursive

echo RUNNING ft.py
/workspace/venv/bin/python3 ft.py
echo DONE merge.py
echo

export QUESTION="Pouvez‑vous lister les options exclusives à la version mobile ?"
echo RUNNING query.py
/workspace/venv/bin/python3 query.py --user-prompt "$QUESTION"
echo DONE query.py
echo

for i in gpt-oss-20b-lora/checkpoint*
do
echo RUNNING query.py on checkpoint $i
/workspace/venv/bin/python query.py --model-name $i --user-prompt "$QUESTION"
echo DONE query.py on checkpoint $i
echo
done

echo RUNNING merge.py
/workspace/venv/bin/python3 merge.py
echo DONE merge.py
cp gpt-oss-20b-tokenizer/* gpt-oss-20b-merged

( cd TensorRT-Model-Optimizer/examples/gpt-oss ; /workspace/venv/bin/python3 convert_oai_mxfp4_weight_only.py --model_path /workspace/gpt-oss-20b-merged --output_path /workspace/gpt-oss-20b-merged-mxfp4 )
aws s3 rm s3://cnam-models/gpt-oss-20b-merged-mxfp4 --recursive
aws s3 cp gpt-oss-20b-merged-mxfp4 s3://cnam-models/gpt-oss-20b-merged-mxfp4 --recursive

aws s3 rm s3://cnam-models/gpt-oss-20b-lora --recursive
aws s3 cp gpt-oss-20b-merged-mxfp4 s3://cnam-models/gpt-oss-20b-lora --recursive
