#!/bin/bash

openssl enc -d -aes-256-cbc -out ft.py -in ft.py.enc -K $mykey -iv $iv
openssl enc -d -aes-256-cbc -out /workspace/.aws/credentials -in /workspace/.aws/credentials.enc -K $mykey -iv $iv

aws s3 cp s3://cnam-models/gpt-oss-20b gpt-oss-20b --recursive

/workspace/venv/bin/python3 ft.py
/workspace/venv/bin/python3 merge.py
cp gpt-oss-20b-tokenizer/* gpt-oss-20b-merged
( cd TensorRT-Model-Optimizer/examples/gpt-oss ; python convert_oai_mxfp4_weight_only.py --model_path /workspace/gpt-oss-20b-merged --output_path /workspace/gpt-oss-20b-merged-mxfp4 )
aws s3 rm s3://cnam-models/gpt-oss-20b-merged-mxfp4 --recursive
aws s3 cp gpt-oss-20b-merged-mxfp4 s3://cnam-models/gpt-oss-20b-merged-mxfp4 --recursive
