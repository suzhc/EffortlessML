import pandas as pd


class MLData():

  def __init__(self, df:pd.DataFrame, x_cols:list[str], y_col:str):
    self.df = df
    self.x_cols = x_cols
    self.y_col = y_col

  def get_x(self):
    return self.df[self.x_cols]

  def get_y(self):
    return self.df[self.y_col]