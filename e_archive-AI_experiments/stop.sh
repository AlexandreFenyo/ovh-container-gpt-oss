#!/usr/bin/zsh

source env.sh
ovhai job stop $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id')
