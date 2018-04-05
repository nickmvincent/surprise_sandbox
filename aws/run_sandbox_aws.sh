#!/usr/bin/env bash
#
# Runs a single AWS job
#

if [ $# -ne 3 ]; then
    echo "Usage: $0_aws.sh s3://path/to/jobs worker_id num_workers" >&2
    exit 1
fi

set -e
set -x

s3_job_dir="$1"
worker_id="$2"
num_workers="$3"
dirs="results standard_results"
num_parallel_jobs=2

for d in $dirs; do
    [ -d "$d" ] || mkdir -p "$d"
done

aws s3 cp ${s3_job_dir}/jobs.txt .

awk "NR % ${num_workers} == ${worker_id}" ./jobs.txt |
parallel -n ${num_parallel_jobs} bash -c '{}' |
tee log.txt

aws s3 cp ./log.txt ${s3_job_dir}/${worker_id}/log.txt

for d in $dirs; do
    aws s3 sync $d ${s3_job_dir}/${worker_id}/$d
done

