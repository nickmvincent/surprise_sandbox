# Estimating Real World Impact of Performance Metrics

### This is mainly a "scratch" paper markdown doc. Will be rather informal.

## Precision
Precision is probably the easiest metric to obtain real-world intuition about.
### Toy Example with Nice Numbers
Let's start with a fake case.

In all our discussions, let's define the retrieval task as:
retrieve as many relevant items (relevant items have a true rating >= 4) as possible and return them in a top k list.
In implementation, this means the recommender system sorts all the "candidates" (which are actual ratings that have been held out for a given fold) and return the top k items of the sorted list.

Say our recommender system has a nice high precision@10 of 0.9. Since lower k values have higher precisio, maybe the precison@5 is 0.95.
We could describe our system like so:
"When presenting a top-ten list, this system gets 9 out of 10 items right." Or put another way, each user should see about 1 error on average. When run on 1 million users, the recommender system makes only 1 million errors.

Now our recommender system is sabotaged by a hacker, who adds a bug to the code: the code system will randomly give out bad ratings.
`if random_number == 17: return 3`
Now the precision@10 drops to 0.8.
"When presenting a top-ten list, this system gets 8 out of 10 items right." For 1 million users, there would be 2 million errors.

So even though we see a -11% change from the hack, this -11% corresponds to a doubling in the error rate.
Unforutunately, to take this further we must use rough estimations or perhaps a user study of some sort. If we assume users have some error rate threshold at which they'll start quitting, it's possible this change pushes past that threshold and has a large effect. However, given that this systems require a lot of buy-in, my involve social entrenchment, etc. it's possible wouldn't cross the threshold.


### Looking at our real numbers
Full dataset precision@10 is 
SVD: 0.8026069529047541
KNN: 0.7999920231446164
w/ 3000 users boycotting:
KNN: 0.785440336
SVD: 0.784764938

