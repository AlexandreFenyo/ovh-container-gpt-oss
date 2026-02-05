#!/bin/bash

openssl enc -d -aes-256-cbc -out ft.py -in ft.py.enc -K $mykey -iv $iv
openssl enc -d -aes-256-cbc -out /workspace/.aws/credentials -in /workspace/.aws/credentials.enc -K $mykey -iv $iv

aws s3 cp s3://cnam-models/gpt-oss-20b gpt-oss-20b --recursive

/workspace/venv/bin/python3 ft.py
