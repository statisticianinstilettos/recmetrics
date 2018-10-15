# recommender_metrics
This library contains useful metrics and plots for evalauting recommender systems.

The python notebook in this repo, `example.ipynb`, contains examples of these plots and metrics in action for a simple popularity recommender. 

## Long Tail Plot
The Long Tail plot is used to explore popularity patterns in user-item interaction data.
<img src="images/long_tail.png" alt="Long Tail Plot" width=400>

## Mar@K
Mean Average Recall at K (Mar@k) measures the recall at the kth recommendations. Mar@k considers the order of recommendations, and penalizes correct recommendations if based on the order of the recommendations. This metric is ideal for evaluating an ordered list of recommendations. Mean Average Precision at K (Map@k) is available in [here](https://github.com/benhamner/Metrics). 

## Coverage
Coverage is the percent of items that the recommender is able to recommend. 

<img src="images/coverage.gif" alt="Coverage" width=400>
Where `I` is the number of items the model recommends in the test data, and `N` is the total number of items in the training data.


## Diversity

## Serendipity

# WIP!
This repo is a work in progress. I will be continually adding metrics as I find them useful for evaluating recommendations. 

<img src="https://media.giphy.com/media/YAnpMSHcurJVS/giphy.gif" width=150>

