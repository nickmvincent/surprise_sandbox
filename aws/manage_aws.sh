#!/usr/bin/env bash
#
# Launch remote ec2 execution of the databoycott simulations.
#

if [ $# -ne "3" ]; then
    echo "usage: $0 secret_access_key s3://job_dir num_workers" >&2
    exit 1
fi

secret_access_key=$1
s3_job_dir=$2
num_workers=$3



for worker_id in $(seq 0 $((num_workers - 1))); do
    echo "doing worker $worker_id"

    cp -p ./aws/bootstrap_aws.sh ./aws/custom_bootstrap.sh
    sed -i '' "s/S3_JOB_DIR/${s3_job_dir}/g" ./aws/custom_bootstrap.sh
    sed -i '' "s/WORKER_ID/${worker_id}/g" ./aws/custom_bootstrap.sh
    sed -i '' "s/NUM_WORKERS/${num_workers}/g" ./aws/custom_bootstrap.sh
    sed -i '' "s#SECRET_ACCESS_KEY#${secret_access_key}#g" ./aws/custom_bootstrap.sh


#   TODO: adapt this to the AWS cli.
#    ec2-run-instances ami-76817c1e \
#        --region us-east-1  \
#        --key shilads-aws-keypair \
#        --user-data-file ${dir}/scrape_citations.sh \
#        --instance-type t2.micro \
#        --subnet subnet-18171730 \
#        --iam-profile myRole \
#        --associate-public-ip-address true \
#        --instance-initiated-shutdown-behavior terminate \
#        --instance-count 95 \
#        --instance-type t2.micro  ||
#            { echo "spot instance request failed!" >&2; exit 1; }
#

done



