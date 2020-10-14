from recmetrics import metrics

import unittest
from unittest import mock


class TestMetrics(unittest.TestCase):

    # def __init__(self, *args, **kwargs):
    #     super(TestMetrics, self).__init__(*args, **kwargs)
    #     self.recsys = Metrics()

    # BUG: Test failing
    def test_novelty(self):
        """
        """
        novelty_score = metrics.novelty(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            pop = {1198: 893, 1270: 876, 593: 876, 2762: 867},
            u = 5,
            n = 3
        )

        self.assertEqual(novelty_score, (2, 3))

    def test_prediction_coverage(self):
        """
        """
        prediction_coverage = metrics.prediction_coverage(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            catalog=['A', 'B', 'C', 'X', 'Y', 'Z'])

        self.assertEqual(prediction_coverage, 50.)

    def test_catalog_coverage(self):
        """
        """
        catalog_coverage = metrics.catalog_coverage(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            catalog=['A', 'B', 'C', 'X', 'Y', 'Z'],
            k=3        
        )

        self.assertEqual(catalog_coverage, 50.)

    def test_mark(self):
        """
        """
        mean_abs_recall_k = metrics.mark(
            actual=[['A', 'B', 'X'], ['A', 'B', 'Y']],
            predicted=[['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            k=5
        )

        self.assertEqual(mean_abs_recall_k, 0.25)

    def test_personalization(self):
        """
        """

        example_predictions = [
            ['1', '2', 'C', 'D'],
            ['4', '3', 'm', 'X'],
            ['7', 'B', 't', 'X']
        ]
        
        personalization_score = metrics.personalization(
            predicted=example_predictions)

        self.assertAlmostEqual(personalization_score, 0.916, places=2)

    def test_intra_list_similarity(self):
        """
        """
        pass

    def test_mse(self):
        """
        """

        y_pred = [1.5, 3.2, 1.4, 3.3, 1.7, 3.8, 4.4, 1.6, 1.2, 3.7]
        
        y_test = [2, 4, 3, 1, 3, 4, 5, 2, 1, 4]

        mse = metrics.mse(y=y_test, yhat=y_pred)

        self.assertAlmostEqual(mse, 1.11, places=2)

    def test_rmse(self):
        """
        """
        
        y_pred = [1.5, 3.2, 1.4, 3.3, 1.7, 3.8, 4.4, 1.6, 1.2, 3.7]
        
        y_test = [2, 4, 3, 1, 3, 4, 5, 2, 1, 4]

        rmse = metrics.rmse(y=y_test, yhat=y_pred)

        self.assertAlmostEqual(rmse, 1.054, places=2)

    # TODO: Additional test coverage for confusion matrix
    @mock.patch("%s.metrics.plt" % __name__)
    def test_make_confusion_matrix(self, mock_plt):
        """
        """
        
        y_pred = [1, 3, 1, 3, 1, 3, 4, 1, 1, 3]
        
        y_test = [2, 4, 3, 1, 3, 4, 5, 2, 1, 4]

        metrics.make_confusion_matrix(
            y = y_test,
            yhat = y_pred
        )

        self.assertTrue(mock_plt.title.called)

    def test_recommender_precision(self):
        """
        """
        recommender_precision = metrics.recommender_precision(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            actual=['A', 'B', 'C', 'X', 'Y', 'Z'])

        self.assertEqual(recommender_precision, 0.)

    def test_recommender_recall(self):
        """
        """
        recommender_recall = metrics.recommender_recall(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            actual=['A', 'B', 'C', 'X', 'Y', 'Z'])

        self.assertEqual(recommender_recall, 0.)
