import unittest
from unittest import mock

import pandas as pd
from recmetrics import metrics


class TestMetrics(unittest.TestCase):

    def test_novelty(self):
        """
        Test novelty function
        """

        # GIVEN test_novelty metrics
        test_predicted_popularity = [[1198, 1270, 593, 2762, 318, 2571, 260, 1240, 296, 608],
                                     [1198, 1270, 593, 2762, 318, 2571, 260, 1240, 296, 608]]

        test_pop = {1198: 893,
                    1270: 876,
                    593: 876,
                    2762: 867,
                    318: 864,
                    2571: 863,
                    260: 859,
                    1240: 857,
                    296: 856,
                    608: 853}

        test_users = 933 # Total unique users from notebook demo
        test_recs_per_user = 10

        # WHEN metrics.novelty is run
        novelty_score, _ = metrics.novelty(
            predicted = test_predicted_popularity,
            pop = test_pop,
            u = test_users,
            n = test_recs_per_user
        )

        # THEN the novelty score should equal the expected value within 3 decimal places
        self.assertEqual(novelty_score, 0.10697151566593581, 3)

    def test_prediction_coverage(self):
        """
        Test prediction_coverage function
        """

        # GIVEN predictions and coverage lists
        test_predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        test_catalog = ['A', 'B', 'C', 'X', 'Y', 'Z']

        # WHEN metrics.prediction_coverage is run
        prediction_coverage = metrics.prediction_coverage(
            predicted = test_predicted,
            catalog = test_catalog)

        # THEN the prediction coverage should equal an expected value of 50
        self.assertEqual(prediction_coverage, 50.)

    def test_catalog_coverage(self):
        """
        Test catalog_coverage function
        """

        # GIVEN test catalog_coverage metrics
        test_predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        test_catalog = ['A', 'B', 'C', 'X', 'Y', 'Z']
        test_k = 3

        # WHEN metrics.catalog_coverage is run
        catalog_coverage = metrics.catalog_coverage(
            predicted = test_predicted,
            catalog = test_catalog,
            k = test_k
        )

        # THEN the coverage should equal 50.
        self.assertEqual(catalog_coverage, 50.)

    def test_mark(self):
        """
        Test mean absolute recall @ k (MAPK) function
        """

        # GIVEN test MAR@K metrics
        test_actual = [['A', 'B', 'X'], ['A', 'B', 'Y']]
        test_predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        test_k = 5

        # WHEN metrics.mark is run
        mean_abs_recall_k = metrics.mark(
            actual=test_actual,
            predicted=test_predicted,
            k=test_k
        )

        # THEN the mean absolute recall @ k should equal the expected value
        self.assertEqual(mean_abs_recall_k, 0.25)

    def test_personalization(self):
        """
        Test personalization function
        """

        # GIVEN a list of lists of predictions
        test_predictions = [
            ['1', '2', 'C', 'D'],
            ['4', '3', 'm', 'X'],
            ['7', 'B', 't', 'X']
        ]
        
        # WHEN metrics.personalization is run
        personalization_score = metrics.personalization(
            predicted=test_predictions)

        # THEN the personalization score should be within the expected value
        self.assertAlmostEqual(personalization_score, 0.916, places=2)

    def test_intra_list_similarity(self):
        """
        Test intra_list_similarity function
        """

        # GIVEN test predictions and a feature matrix (formatted as a DataFrame)

        test_predictions = [
            [3, 7, 5, 9],
        ]

        lst_features = {1: {'Action': 0, 'Comedy': 0, 'Romance': 0},
                        2: {'Action': 0, 'Comedy': 0, 'Romance': 0},
                        3: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                        4: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                        5: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                        6: {'Action': 1, 'Comedy': 0, 'Romance': 0},
                        7: {'Action': 0, 'Comedy': 1, 'Romance': 0},
                        8: {'Action': 0, 'Comedy': 0, 'Romance': 0},
                        9: {'Action': 1, 'Comedy': 0, 'Romance': 0},
                        10: {'Action': 1, 'Comedy': 0, 'Romance': 0}}

        feature_df = pd.DataFrame.from_dict(lst_features, orient="index").reset_index().rename(
            columns={"index": "movieId"}).set_index("movieId")

        # WHEN metrics.intra_list_similarity is run
        intra_list_similarity = metrics.intra_list_similarity(
            test_predictions, feature_df
        )

        # THEN the expected value should be 0.5 within 3 decimal places
        self.assertAlmostEqual(intra_list_similarity, 0.5, places=3)

    def test_mse(self):
        """
        Test mean squared error (MSE) function
        """

        # GIVEN predictions and actual values
        y_pred = [1.5, 3.2, 1.4, 3.3, 1.7, 3.8, 4.4, 1.6, 1.2, 3.7]
        y_test = [2, 4, 3, 1, 3, 4, 5, 2, 1, 4]

        # WHEN metrics.mse is run
        mse = metrics.mse(y=y_test, yhat=y_pred)

        # THEN the expected MSE should be 1.11 within two decimal places
        self.assertAlmostEqual(mse, 1.11, places=2)

    def test_rmse(self):
        """
        Test root mean square error (RMSE) function
        """
        
        # GIVEN predictions and actual values
        y_pred = [1.5, 3.2, 1.4, 3.3, 1.7, 3.8, 4.4, 1.6, 1.2, 3.7]
        y_test = [2, 4, 3, 1, 3, 4, 5, 2, 1, 4]

        # WHEN metrics.rmse is run
        rmse = metrics.rmse(y=y_test, yhat=y_pred)

        # THEN the expected RMSE should be 1.054 within two decimal places
        self.assertAlmostEqual(rmse, 1.054, places=2)

    @mock.patch("%s.metrics.plt" % __name__)
    def test_make_confusion_matrix(self, mock_plt):
        """
        Test make_confusion_matrix function

        This test assumes the plot output is correct
        """
        
        # GIVEN predictions and actual values
        y_pred = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0]
        y_test = [0, 1, 0, 0, 0, 0, 1, 1, 0, 0]

        # WHEN metrics.make_confusion_matrix is run
        metrics.make_confusion_matrix(
            y = y_test,
            yhat = y_pred
        )

        # THEN plt.title() and plt.show() should be called in the function
        self.assertTrue(mock_plt.title.called)
        self.assertTrue(mock_plt.show.called)

    def test_recommender_precision(self):
        """
        Test recommender_precision function
        """

        # GIVEN predictions and actual values
        test_predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        test_actual = [['A', 'B', 'X'], ['A', 'B', 'Y']]
        
        # WHEN metrics.recommender_precision is run
        recommender_precision = metrics.recommender_precision(
            predicted = test_predicted,
            actual = test_actual)

        # THEN the expected value should equal 0.333 within three decimal places
        self.assertAlmostEqual(recommender_precision, 0.333, places=3)

    def test_recommender_recall(self):
        """
        Test recommender_recall function
        """

        # GIVEN predictions and actual values
        test_predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
        test_actual = [['A', 'B', 'X'], ['A', 'B', 'Y']]

        # WHEN metrics.recommender_recall is run
        recommender_recall = metrics.recommender_recall(
            predicted = test_predicted,
            actual= test_actual)

        # THEN the expected value should equal 0.333 within three decimal places
        self.assertAlmostEqual(recommender_recall, 0.333, places=3)
