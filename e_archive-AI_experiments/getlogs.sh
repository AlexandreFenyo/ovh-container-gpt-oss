#!/usr/bin/zsh

source env.sh
ovhai job log $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id')
