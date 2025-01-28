from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import training_results, validation_results
from matchpredictor.predictors.sophiscated_predictor import train_sophiscated_predictor
from test.predictors import csv_location

# I do not know if it is allowed but i copied and pasted your linear regression model to start with
# the steps to get training data and validation data are the same; we can play with the filters though
# the evaluation step and assertion are the same; we can play with the parameters
class TestSophiscatedPredictor(TestCase):
    def test_accuracy(self) -> None:
        training_data = training_results(csv_location, 2019, result_filter=lambda result: result.season >= 2016)
        validation_data = validation_results(csv_location, 2019)
        predictor = train_sophiscated_predictor(training_data)

        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        self.assertGreaterEqual(accuracy, .33)
