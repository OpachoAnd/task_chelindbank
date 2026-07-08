import pandas as pd


class MacroCategorySearch:
    """
    Класс для поиска топ-5 кластеров 
    """
    def __init__(self):
        self.stat_df = pd.DataFrame()

    def variance_trans_amount(self, df: pd.DataFrame):
        self.stat_df = df.var()
        print('variance_series\n', pd.DataFrame(self.stat_df))
