import pandas as pd
import numpy as np
from tabulate import tabulate

from EffortlessML.data import MLData
from EffortlessML.auto_ml import PycaretAutoML
from EffortlessML.exp import ExpML


def ml(df: pd.DataFrame, x_cols: list[str], y_col):
    data = MLData(df=df, x_cols=x_cols, y_col=y_col)
    data.report()
    auto_ml = PycaretAutoML(df_data=data.df, y_col=data.y_col, x_cols=data.x_cols)
    report = auto_ml.generate_report()
    print('模型比较结果：')
    print(tabulate(report.drop(columns='Model'), headers='keys', tablefmt='pretty'))
    auto_ml.tune_model()
    print('模型参数：')
    print(auto_ml.best_model)
    print('五折交叉验证结果：')
    exp = ExpML(X=data.get_x(), y=data.get_y(), model_name=auto_ml.best_model)
    exp.plot_roc()
    exp.plot_cm()
    auto_ml.plot_feature_importance()
    auto_ml.plot_shap()
    