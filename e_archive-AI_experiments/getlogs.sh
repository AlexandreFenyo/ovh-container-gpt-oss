#!/usr/bin/zsh

ovhai job log $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id')
