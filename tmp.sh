s3_job_dir="s3://abc/def"
parent=$(dirname -- "$s3_job_dir")
final="${parent}/predictions/standards"
echo ${final}
