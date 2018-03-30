#!/usr/bin/env bash
#
# Launch remote ec2 execution of the databoycott simulations.
#

if [ $# -ne "3" ]; then
    echo "usage: $0 secret_access_key s3://job_dir num_workers" >&2
    exit 1
fi


set -e
set -x

secret_access_key=$1
s3_job_dir=$2
num_workers=$3

for worker_id in $(seq 0 $((num_workers - 1))); do
    echo "doing worker $worker_id"

    cp -p ./aws/bootstrap_aws.sh ./aws/custom_bootstrap.sh
    sed -i '' "s#S3_JOB_DIR#${s3_job_dir}#g" ./aws/custom_bootstrap.sh
    sed -i '' "s/WORKER_ID/${worker_id}/g" ./aws/custom_bootstrap.sh
    sed -i '' "s/NUM_WORKERS/${num_workers}/g" ./aws/custom_bootstrap.sh
    sed -i '' "s#SECRET_ACCESS_KEY#${secret_access_key}#g" ./aws/custom_bootstrap.sh

    userdata="$(cat ./aws/custom_bootstrap.sh | base64 | tr -d '\n' )"
    cp -p ./aws/launch_specification.json ./aws/launch_specification_custom.json
    sed -i '' "s/USER_DATA/${userdata}/g" ./aws/launch_specification_custom.json

    aws ec2 request-spot-instances \
        --valid-until "2018-05-06T02:52:51.000Z" \
        --instance-interruption-behavior terminate \
        --type one-time \
        --instance-count 1 \
        --spot-price "2.00" \
        --launch-specification "file://aws/launch_specification_custom.json"

done



