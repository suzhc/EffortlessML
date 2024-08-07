import pycaret.classification as caret_c
from h2o.automl import H2OAutoML
import h2o
import pandas as pd


class AutoML:
    def __init__(self, df_data: pd.DataFrame, y_col: str, x_cols: list[str]):
        self.df_data = df_data
        self.y_col = y_col
        self.x_cols = x_cols

    def generate_report(self):
        pass


class PycaretAutoML(AutoML):
    def __init__(self, df_data: pd.DataFrame, y_col: str, x_cols: list[str]):
        super().__init__(df_data, y_col, x_cols)
        df = self.df_data[self.x_cols + [self.y_col]]
        self.s = caret_c.setup(
            data=df, target=self.y_col, train_size=0.8, session_id=123, verbose=False
        )
        self.best_model = caret_c.compare_models(fold=5, sort='f1', verbose=False)
    
    def generate_report(self):
        lb = caret_c.get_leaderboard(verbose=False)
        lb = lb.sort_values(by='F1', ascending=False)\
                .reset_index()\
                .drop(columns='Index')
        return lb
    
    def plot_feature_importance(self):
        caret_c.plot_model(self.best_model, plot='feature')
    
    def plot_shap(self):
        caret_c.interpret_model(self.best_model)
    
    def tune_model(self):
        self.best_model = caret_c.tune_model(self.best_model, verbose=False)


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
