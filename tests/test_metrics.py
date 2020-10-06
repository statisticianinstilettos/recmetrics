from recmetrics import metrics

import unittest


class TestMetrics(unittest.TestCase):

    # def __init__(self, *args, **kwargs):
    #     super(TestMetrics, self).__init__(*args, **kwargs)
    #     self.recsys = Metrics()

    # BUG: Test failing
    def test_novelty(self):

        COV = metrics.novelty(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            pop = {1198: 893, 1270: 876, 593: 876, 2762: 867},
            u = 5,
            n = 3
        )

        self.assertEqual(COV, (2, 3))

    def test_prediction_coverage(self):

        COV = metrics.prediction_coverage(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            catalog=['A', 'B', 'C', 'X', 'Y', 'Z'])

        self.assertEqual(COV, 50.)

    def test_catalog_coverage(self):

        COV = metrics.catalog_coverage(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            catalog=['A', 'B', 'C', 'X', 'Y', 'Z'],
            k=3        
        )

        self.assertEqual(COV, 50.)

    # TODO: Create test
    def test_ark(self):
        # metrics._ark()
        pass

    def test_mark(self):

        COV = metrics.mark(
            actual=[['A', 'B', 'X'], ['A', 'B', 'Y']],
            predicted=[['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            k=5
        )

        self.assertEqual(COV, 0.25)

    # BUG: Test failing
    def test_personalization(self):

        COV = metrics.personalization(
            predicted=[['X', 'Y', 'Z'], ['X', 'Y', 'Z']])

        self.assertAlmostEqual(COV, -2.22, places=2)

    def test_single_list_similarity(self):
        pass

    def test_intra_list_similarity(self):
        pass

    def test_mse(self):

        y_pred = [1.5, 3.2, 1.4, 3.3, 1.7, 3.8, 4.4, 1.6, 1.2, 3.7]
        
        y_test = [2, 4, 3, 1, 3, 4, 5, 2, 1, 4]

        mse = metrics.mse(y=y_test, yhat=y_pred)

        self.assertAlmostEqual(mse, 1.11, places=2)

    def test_rmse(self):
        
        y_pred = [1.5, 3.2, 1.4, 3.3, 1.7, 3.8, 4.4, 1.6, 1.2, 3.7]
        
        y_test = [2, 4, 3, 1, 3, 4, 5, 2, 1, 4]

        rmse = metrics.rmse(y=y_test, yhat=y_pred)

        self.assertAlmostEqual(rmse, 1.054, places=2)

    def test_make_confusion_matrix(self):
        pass

    def test_recommender_precision(self):
        COV = metrics.recommender_precision(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            actual=['A', 'B', 'C', 'X', 'Y', 'Z'])

        self.assertEqual(COV, 0.)


    def test_recommender_recall(self):
        COV = metrics.recommender_recall(
            predicted = [['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            actual=['A', 'B', 'C', 'X', 'Y', 'Z'])

        self.assertEqual(COV, 0.)
