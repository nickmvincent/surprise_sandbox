# Surprise Sandbox

Able to run a variety of recsys tests using Surprise.

Requires Surprise to be installed.



Pipeline:

Get files in places with your choice of:
cli, finder app, aws cli, etc

Merge standards files
`python .\standards_for_uid_sets.py --join`

Do the processing (i.e. match up columns and do substraction)
`python process_all.py`

Re-run visualization and/or statistics