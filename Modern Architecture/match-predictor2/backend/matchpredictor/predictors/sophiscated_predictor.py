from typing import List, Tuple, Optional, cast

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction
from sklearn.tree import DecisionTreeClassifier
# I used your linear regression model as a template
class SophiscatedPredictor(Predictor):
    def __init__(self, model: DecisionTreeClassifier, encoder: OneHotEncoder) -> None:
        self.model = model

    def predict(self, fixture: Fixture) -> Prediction:
        encodedX=self.encoder.transform([[fixture.home_team.name,fixture.away_team.name]])
        # becuz of handle_unknown = ignore i dont need to deal wtih exceptions here
        
        pred = self.model.predict(encodedX)

        # well i use your code I think it makes sense.. I can reverse neg and pos though to show you that I understand the code 
        if pred > 0:
            return Prediction(outcome=Outcome.AWAY)
        elif pred < 0:
            return Prediction(outcome=Outcome.HOME)
        else:
            return Prediction(outcome=Outcome.DRAW)

def train_sophiscated_predictor(results: List[Result]) -> Predictor:
    y=np.sign(np.array([r.away_goals for r in results])-np.array([r.home_goals for r in results]))
    home = np.array([r.fixture.home_team.name for r in results]).reshape(-1, 1)
    away = np.array([r.fixture.away_team.name for r in results]).reshape(-1, 1)
    x = np.concatenate([home,away], 1)
    print(x.shape)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')# to avoid errors from new teams, i use handle_unknown=ignore
    encoded_x = encoder.fit_transform(x)
    model = DecisionTreeClassifier()
    model.fit(encoded_x, y)
    return SophiscatedPredictor(model,encoder)
