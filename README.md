# recmetrics
This library contains useful diagnostic metrics and plots for evaluating recommender systems.

The python notebook in this repo, `example.ipynb`, contains examples of these plots and metrics in action using the [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/).

Install with `pip install recmetrics`

For instructions on how to set up your own python package using twine, check out https://pypi.org/project/twine/


## Long Tail Plot
The Long Tail plot is used to explore popularity patterns in user-item interaction data. Typically, a small number of items will make up most of the volume of interactions and this is referred to as the "head". The "long tail" typically consists of most products, but make up a small percent of interaction volume.

<img src="images/long_tail_plot.png" alt="Long Tail Plot" width=600>

The items in the "long tail" usually do not have enough interactions to accurately be recommended using user-based recommender systems like collaborative filtering due to inherent popularity bias in these models and data sparsity. Many recommender systems require a certain level of sparsity to train. A good recommender must balance sparsity requirements with popularity bias.

## Mar@K and Map@K
Mean Average Recall at K (Mar@k) measures the recall at the kth recommendations. Mar@k considers the order of recommendations, and penalizes correct recommendations if based on the order of the recommendations. Map@k and Mar@k are ideal for evaluating an ordered list of recommendations. There is a fantastic implmentation of Mean Average Precision at K (Map@k) available [here](https://github.com/benhamner/Metrics), so I have not included it in this repo.

<img src="images/mark_plot.png" alt="Mar@k" width=600>
Map@k and Mar@k metrics suffer from popularity bias. If a model works well on popular items, the majority of recommendations will be correct, and Mar@k and Map@k can appear to be high while the model may not be making useful or personalized recommendations.

## Coverage
Coverage is the percent of items that the recommender is able to recommend.

<img src="images/coverage_equation.gif" alt="Coverage Equation" width=200>
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

## MSE and RMSE
Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are used to evaluate the accuracy of predicted values yhat such as ratings compared to the true value, y.
These can also be used to evalaute the reconstruction of a ratings matrix.

<img src="images/mse.gif" alt="MSE Equation" width=200>

<img src="images/rmse.gif" alt="RMSE Equation" width=200>

## Predicted Class Probability Distribution Plots
This plot shows the distributions of class prediction probabilities from a binary classification model. This plot is typically used to visualize how well a model is able to separate the classes, and to pick the optimal decision threshold to classify observations to class 1 (0.5 is usually the default threshold for this method). This plot could also be used to visualize the recommendation scores. For example, the predicted 5 star rating for a single movie could be visualized with this plot to see the optimal predicted rating score to threshold into a prediction of that movie.  

<img src="images/class_probs.png" alt="RMSE Equation" width=400>

## ROC and AUC
The Receiver Operating Characteristic (ROC) plot is used to visualize the tradeoff between true positives and false positives for binary classification. The Area Under the Curve (AUC) is sometimes used as an evaluation metrics. 

<img src="images/ROC.png" alt="RMSE Equation" width=600>

## Precision and Recall Curve
The Precision and Recall plot is used to visualize the tradeoff between precision and recall for one class in a classification.
<img src="images/PrecisionRecallCurve.png" alt="RMSE Equation" width=200>

## Confusion Matrix
coming soon...

## Rank Order Analysis
coming soon...

## How to create a python package with PyPi and twine
https://pypi.org/project/twine/

# WIP!
This repo is a work in progress. I am continually adding metrics as I find them useful for evaluating recommendations. I you would like to contribute, please contact me and I'll give you access to make a branch. longoclaire@gmail.com :-)

<img src="https://media.giphy.com/media/YAnpMSHcurJVS/giphy.gif" width=400>

