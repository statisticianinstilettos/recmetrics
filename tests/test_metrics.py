from recmetrics import metrics

import unittest


class TestMetrics(unittest.TestCase):

    # def __init__(self, *args, **kwargs):
    #     super(TestMetrics, self).__init__(*args, **kwargs)
    #     self.recsys = Metrics()

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
    
    def test_mark(self):

        COV = metrics.mark(
            actual=[['A', 'B', 'X'], ['A', 'B', 'Y']],
            predicted=[['X', 'Y', 'Z'], ['X', 'Y', 'Z']],
            k=5
        )

        self.assertEqual(COV, 0.25)

    def test_personalization(self):

        COV = metrics.personalization(
            predicted=[['X', 'Y', 'Z'], ['X', 'Y', 'Z']])

        self.assertAlmostEqual(COV, -2.220446, places=4)

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
