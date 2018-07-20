# On efficiency and ease-of-understanding
## 6/18/2018
Right now, the code is "mostly" efficient (with one large issue), but very confusing to actually run.

### Efficiency issues
Right now that standard values are _not computed efficiently_ because we don't save the actual recommender system predictions. Instead we regenerate the predictions for each test. While we do batch quite a few tests at once, given memory constraints, it would be even faster to just _save the predictions_. In other words, for our boycott conditions, we should never need to train each recommender algorithm more than one time, ever. Similarly, saving predictions for other experiments would be helpful for deeper analysis of our results.



The whole data processing pipeline for the CSCW submission was quite manual.
It requires that a human specify carefully all the experiments, but then also make sure that the "no boycott standard results" are specified as well, and specified properly. There is room for human error here, e.g. if you forget an experiment it might not show up in the final results. Luckily we can check for this human error by inspecting the final results and looking for missing data points.

However, the organization of files is still quite confusing and there's a lot of potential confusing redundancy in the files, mainly due to the fact that each AWS spot instance is a full clone of the project directory. For example, this means that the directories with standard results (currently in `rand_standards`) each have a copy of every `uid_sets` doc, even those these are unused. Future versions should clean out this redundant data to make human error harder.


## Things to double check
One important bit of code to double-check is that the boycott and like-boycott testing is implemented correctly.
i.e. make sure the like-boycott users are actually properly included in the training


# 7/19
Switched to using SVD with random state 0 to make it even more reproducible