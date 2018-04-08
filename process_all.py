"""
This is just a convenience script
that runs process_results with a bunch
of saved params.

Note: this DOES not run process_results on every file in the results/ dir
It only runs the files specified here!

So if something isn't being processed, you could edit this file
"""

import os

os.system("python process_results.py --grouping gender --userfracs 0.5,1 --ratingfracs 0.5,1")  