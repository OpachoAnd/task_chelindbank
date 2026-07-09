import pandas as pd

class FeatureEngineering:
    def __init__(self):
        ...

    def new_feature(self, df: pd.DataFrame, name_col_income: str):
        """
        Метод готовит новые фичи для последующей кластеризации

        name_col_income - имя колонки с общим доходом
        """

        df_feature = df.copy(deep=True)
        df_feature[name_col_income] = df_feature.sum(axis=1)
        print('df_feature\n', df_feature)
    


