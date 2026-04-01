#!/usr/bin/zsh

cd e:/AI_experiments
ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id' | xargs ovhai job delete
while [ ! -z $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id') ]
do
    echo removing latest job with same name
    sleep 5
done
aws s3 cp runs/"$RUN"/params.cfg s3://cnam-models/runs/"$RUN"/params.cfg
ovhai job run --name "$RUN" --flavor h100-1-gpu --gpu 1 --ssh-public-keys $sshkey --unsecure-http -e mykey=$mykey -e iv=$iv -e wandbkey=$wandbkey -e hfkey=$hfkey -e cfg=s3://cnam-models/runs/"$RUN"/params.cfg fenyoa/ft_gpt_oss_20b_ovh_faq
echo job launched:
ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .spec.name'
export JOBID=$(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id')
echo job ID: $JOBID
while [ $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .status.info.code') != JOB_DONE ]
do
  sleep 5 ; date ; echo retry ; ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .status.info.code'
done
ovhai job log $(ovhai job list -a --output json | jq -r '.[] | select(.spec.name=="'$RUN'") | .id') > runs/"$RUN"/job.log
( cd runs/"$RUN" ; "/cygdrive/c/Program Files/Git/cmd/git" clone https://huggingface.co/datasets/$(egrep '^var_dataset_name="' params.cfg | sed 's/^var_dataset_name="//' | sed 's/"$//') )
rm -rf runs/"$RUN"/job_output
aws s3 cp s3://cnam-models/job_output runs/"$RUN"/job_output --recursive
rm -rf runs/"$RUN"/gpt-oss-20b-lora
aws s3 cp s3://cnam-models/gpt-oss-20b-lora runs/"$RUN"/gpt-oss-20b-lora --recursive
rm -rf /mnt/e/gpt-oss-20b-merged-mxfp4
cd e:/gpt-oss-20b-merged-mxfp4
aws s3 cp s3://cnam-models/gpt-oss-20b-merged-mxfp4 . --recursive
ollama show gpt-oss:20b --modelfile > Modelfile.tmpl
echo 'SYSTEM """' > Modelfile
grep system_prompt e:/AI_experiments/runs/"$RUN"/params.cfg | sed 's/^system_prompt="//' | sed 's/"$//' >> Modelfile
echo 'SYSTEM """' >> Modelfile
cat Modelfile.tmpl | sed 's/^FROM D:.*/FROM E:\\gpt-oss-20b-merged-mxfp4/' >> Modelfile
ollama create "$RUN":20b -f Modelfile
ollama run "$RUN":20b "Qui es-tu ?"
