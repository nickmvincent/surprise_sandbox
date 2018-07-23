#!/usr/bin/env bash
#
# Script to bootstrap an AWS environment and launch a simulation run.
# Note that this script will be downloaded independently and run in a
# "clean" AWS environment so it must install any external dependencies
# it needs.
#
# m5.2xlarge
#
# Usage: ./bootstrap_aws.sh secret_access_key s3://path/to/jobs worker_id num_workers
#

# These parameters get rewritten by the manage_aws.sh script.
secret_access_key=SECRET_ACCESS_KEY
s3_job_dir=S3_JOB_DIR
worker_id=WORKER_ID
num_workers=NUM_WORKERS

# fail on error and enable logging
set -e
set -x

cd /root

# install
yum update -y
yum install -y git gcc gcc-c++ parallel
yum install -y atlas-devel lapack-devel blas-devel libgfortran
yum install -y python36 python36-devel

pip-3.6 install Cython numpy scipy pandas joblib psutil

# Install custom surprise
git clone https://github.com/nickmvincent/Surprise.git
cd Surprise/
python3 ./setup.py install
cd ..

# Configure aws
mkdir .aws
cat >.aws/credentials << EOF
[default]
aws_access_key_id = AKIAJHG4JDWYRJITSUMQ
aws_secret_access_key = ${secret_access_key}
EOF
echo "[default]" >.aws/config

git clone https://github.com/nickmvincent/surprise_sandbox.git
cd surprise_sandbox

mkdir predictions
cd predictions
mkdir standards
cd standards

parent=$(dirname -- "$s3_job_dir")
final="${parent}/predictions/standards"
aws s3 cp ${final} . --recursive


./aws/run_sandbox_aws.sh "${s3_job_dir}" "${worker_id}" "${num_workers}"


/sbin/halt