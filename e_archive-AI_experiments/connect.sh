#!/usr/bin/zsh

source env.sh
ssh $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id')@gra.ai.cloud.ovh.net
