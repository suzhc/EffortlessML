from pycaret.classification import *
from h2o.automl import H2OAutoML
import h2o
import pandas as pd


class AutoML():
    def __init__(self, df_data: pd.DataFrame, y_col: str, x_cols: list[str]):
        self.df_data = df_data
        self.y_col = y_col
        self.x_cols = x_cols

    def generate_report(self):
        pass


class PycaretAutoML(AutoML):
    def __init__(self, df_data: pd.DataFrame, y_col: str, x_cols: list[str]):
        super().__init__(df_data, y_col, x_cols)

    def generate_report(self):
        s = setup(self.df_data, target = self.y_col, session_id = 123)
        best = compare_models()
        return best


class H2OAutoML(AutoML):
    def __init__(self, df_data: pd.DataFrame, y_col: str, x_cols: list[str]):
        super().__init__(df_data, y_col, x_cols)

    def generate_report(self):

        h2o.init()
        hf_data = h2o.H2OFrame(self.df_data)
        y = self.y_col
        x = self.x_cols
        hf_data[y] = hf_data[y].asfactor()

        aml = H2OAutoML(max_models=50, seed=123)
        aml.train(x=x, y=y, training_frame=hf_data)

        lb = aml.leaderboard
        return lb