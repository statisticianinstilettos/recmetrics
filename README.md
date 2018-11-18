# recommender_metrics
This library contains useful diagnostic metrics and plots for evaluating recommender systems. 

The python notebook in this repo, `example.ipynb`, contains examples of these plots and metrics in action for a simple popularity recommender.

Install with `pip install git+https://github.com/statisticianinstilettos/recmetrics.git#egg=recmetrics`
The metrics and plots class can be imported in python as:

`from recmetrics import metrics`

`from recmetrics import plots`

## Long Tail Plot
The Long Tail plot is used to explore popularity patterns in user-item interaction data. Typically, a small number of items will make up most of the volume of interactions and this is referred to as the "head". The "long tail" typically consists of most products in a catalog, but make up a small percent of interaction volume.
<img src="images/long_tail_plot.png" alt="Long Tail Plot" width=400>

Typically, only a small percentage of products will have a high volume of user-interactions such as clicks, ratings, or purchases. These are called the items in the "head". The items in the "long tail" typically do not have enough interactions to accurately be recommended using user-based recommender systems like collaborative filtering due to inherent popularity bias in these models. However, methods that can recommend long tail items can result in better-personalized recommendations and higher revenue. The items in the "head" are popular items that user's may not have difficulty finding themselves, and it can be argued that recommending these items is not as helpful as recommending an item that would be considered new and relevant to the user.

## Mar@K and Map@K
Mean Average Recall at K (Mar@k) measures the recall at the kth recommendations. Mar@k considers the order of recommendations, and penalizes correct recommendations if based on the order of the recommendations. Map@k and Mar@k are ideal for evaluating an ordered list of recommendations. There is a fantastic implmentation of Mean Average Precision at K (Map@k) available [here](https://github.com/benhamner/Metrics), so I have not included it in this repo.

<img src="images/mark_plot.png" alt="Mar@k" width=400>
Map@k and Mar@k metrics suffer from popularity bias. If a model works well on popular items, the majority of recommendations will be correct, and Mar@k and Map@k can appear to be high while the model may not be making useful or personalized recommendations. 

## Coverage
Coverage is the percent of items that the recommender is able to recommend.

<img src="images/coverage_equation.gif" alt="Coverage Equation" width=400>
Where 'I' is the number of unique items the model recommends in the test data, and 'N' is the total number of unique items in the training data.

<img src="images/coverage_plot.png" alt="Coverage Plot" width=400>

## Personalization
Personalization is the dissimilarity between user's lists of recommendations. 
A high score indicates user's recommendations are different).
A low personalization score indicates user's recommendations are very similar.
    
For example, if two users have recommendations lists [A,B,C,D] and [A,B,C,Y], the personalization can be calculated as:
<img src="images/personalization_code.png" alt="Coverage Plot" width=400>


## Intra-list Similarity
Intra-list similarity uses a feature matrix to calculate the cosine similarity between the items in a list of recommendations.
The feature matrix is indexed by the item id and includes one-hot-encoded features.
If a recommender system is recommending lists of very similar items, the intra-list similarity will be high. 

<img src="images/ils_matrix.png" alt="Coverage Plot" width=400>

<img src="images/ils_code.png" alt="Coverage Plot" width=400>

# WIP!
This repo is a work in progress. I am continually adding metrics as I find them useful for evaluating recommendations.

<img src="https://media.giphy.com/media/YAnpMSHcurJVS/giphy.gif" width=400>
