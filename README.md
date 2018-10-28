# recommender_metrics
This library contains useful diagnostic metrics and plots for evaluating recommender systems. 

The python notebook in this repo, `example.ipynb`, contains examples of these plots and metrics in action for a simple popularity recommender.

## Long Tail Plot
The Long Tail plot is used to explore popularity patterns in user-item interaction data. Typically, a small number of items will make up most of the volume of interactions and this is referred to as the "head". The "long tail" typically consists of most products in a catalog, but make up a small percent of interaction volume.
<img src="images/long_tail_plot.png" alt="Long Tail Plot" width=400>

## Mar@K and Map@K
Mean Average Recall at K (Mar@k) measures the recall at the kth recommendations. Mar@k considers the order of recommendations, and penalizes correct recommendations if based on the order of the recommendations. Map@k and Mar@k are ideal for evaluating an ordered list of recommendations. There is a fantastic implmentation of Mean Average Precision at K (Map@k) available [here](https://github.com/benhamner/Metrics), so I have not included it in this repo.

<img src="images/mark_plot.png" alt="Mar@k" width=400>

## Coverage
Coverage is the percent of items that the recommender is able to recommend.

<img src="images/coverage_equation.gif" alt="Coverage Equation" width=400>
Where 'I' is the number of unique items the model recommends in the test data, and 'N' is the total number of unique items in the training data.

<img src="images/coverage_plot.png" alt="Coverage Plot" width=400>


# WIP!
This repo is a work in progress. I am continually adding metrics as I find them useful for evaluating recommendations.

<img src="https://media.giphy.com/media/YAnpMSHcurJVS/giphy.gif" width=400>
