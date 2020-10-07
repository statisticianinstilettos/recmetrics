import unittest
from unittest import mock

from recmetrics import plots


class TestPlots(unittest.TestCase):

    def test_long_tail_plot(self):
        pass

    # BUG: Test failing
    @mock.patch("%s.plots.sns" % __name__)
    def test_coverage_plot(self, mock_plt):

        COVERAGE_SCORES = [0.17, 0.25, 0.76]

        MODEL_NAMES = ['Model A', 'Model B', 'Model C']

        plots.coverage_plot(coverage_scores=COVERAGE_SCORES, model_names=MODEL_NAMES)

        # Assert plt.title has been called with expected arg
        # mock_plt.ax.set_title.assert_called_once_with("Catalog Coverage in X")

        # Assert plt.figure got called
        # assert mock_plt.ax.set_title.called
        # self.assertTrue(mock_plt.show.called)
        self.assertTrue(mock_plt.barplot.called)

    @mock.patch("%s.plots.sns" % __name__)
    def test_personalization_plot(self, mock_sns):
        
        SCORES = [0.13, 0.52, 0.36]

        MODEL_NAMES = ["Model A", "Model B", "Model C"]

        plots.personalization_plot(personalization_scores=SCORES,
            model_names=MODEL_NAMES)

        # mock_sns.set_title.assert_called_once_with("Personalization")
        
        self.assertTrue(mock_sns.barplot.called)

    def test_intra_list_similarity_plot(self):
        pass

    def test_mark_plot(self):
        pass

    def test_mapk_plot(self):
        pass

    def test_class_separation_plot(self):
        pass

    def test_roc_plot(self):
        pass

    def test_precision_recall_plot(self):
        pass

    def test_make_listy(self):
        pass

    def test_is_listy(self):
        pass

    def test_metrics_plot(self):
        pass
