#!/usr/bin/zsh

ovhai job stop $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id')
