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

secret_access_key="$1"
s3_job_dir="$2"
worker_id="$3"
num_workers="$4"

# fail on error and enable logging
set -e
set -x

# install
yum update
yum install -y git gcc gcc-c++
yum install -y atlas-devel lapack-devel blas-devel libgfortran
yum install -y python36 python36-devel

pip-3.6 install --upgrade pip
pip-3.6 install Cython numpy scipy

# Install custom surprise
git clone https://github.com/nickmvincent/Surprise.git
cd Surprise/
python3 ./setup.py install
cd ..

# Configure aws
mkdir .aws
cat >.aws/credentials << EOF
[default]
aws_access_key_id = AKIAJXMF3MK7C3K2NIFQ
aws_secret_access_key = ${secret_access_key}
EOF
echo "[default]" >.aws/config

# Install our package
git clone https://github.com/nickmvincent/surprise_sandbox.git
cd surprise_sandbox

