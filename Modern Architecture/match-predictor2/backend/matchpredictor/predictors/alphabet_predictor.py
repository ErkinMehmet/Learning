from typing import List, Tuple, Optional, cast

import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore

from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction

# I used your linear regression model as a template
class AlphabetPredictor(Predictor):
    def __init__(self, model: LogisticRegression) -> None:
        self.model = model

    def predict(self, fixture: Fixture) -> Prediction:
        alphaHome = convertTeam(fixture.home_team)
        alphaAway = convertTeam(fixture.away_team)

        if alphaHome is None:
            return Prediction(outcome=Outcome.AWAY)
        if alphaAway is None:
            return Prediction(outcome=Outcome.HOME)
        
        pred = self.model.predict([[alphaHome,alphaAway]])

        # well i use your code I think it makes sense.. I can reverse neg and pos though to show you that I understand the code 
        if pred > 0:
            return Prediction(outcome=Outcome.AWAY)
        elif pred < 0:
            return Prediction(outcome=Outcome.HOME)
        else:
            return Prediction(outcome=Outcome.DRAW)

def convertTeam(team: Team) -> Optional[float64]:
    try:
        #print(team,type(team))
        codes=np.array([ord(char) for char in team.name])
        result:float64=0
        for i in range(len(codes)):
            result+=codes[i]*128**-i # to sort strings in ascending order, the first letter having most weight
        return result
    except ValueError:
        return None


def return_model(results: List[Result]) -> LogisticRegression:
    home = np.array([convertTeam(r.fixture.home_team) for r in results]).reshape(-1, 1)
    away = np.array([convertTeam(r.fixture.away_team) for r in results]).reshape(-1, 1)

    y=np.sign(np.array([r.away_goals for r in results])-np.array([r.home_goals for r in results])) # if pos then away wins, 0 draw, neg then home wins

    x = np.concatenate([home,away], 1)
    print(x.shape,y.shape)
    model = LogisticRegression(solver='liblinear')
    model.fit(x, y)

    return model


def train_alphabet_predictor(results: List[Result]) -> Predictor:
    mo = return_model(results)
    return AlphabetPredictor(mo)
