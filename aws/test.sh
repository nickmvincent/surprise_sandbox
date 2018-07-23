s3_job_dir="s3://abc/def"
parent=$(dirname -- "$s3_job_dir")
echo parent is $parent
final=$parent/predictions/standards
echo final is $final
x=string
echo $x
y=$xplusstuff
echo $y
