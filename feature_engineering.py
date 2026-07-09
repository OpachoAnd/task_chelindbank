import pandas as pd
import numpy as np


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

        # Выбираем столбцы которые нужно поделить на доход
        cols_div = df_feature.columns.drop(name_col_income)

        # Получим столбцы категорий после деления на общий доход 
        df_feature[cols_div] = df_feature[cols_div].div(df_feature[name_col_income].values[:, None])
        df_feature.drop(columns=[name_col_income], inplace=True)
        # Замена значений inf если было деление на 0 (У клиента не было расходов ни в одной категории)
        df_feature.replace([np.inf, -np.inf], 0, inplace=True)
        return df_feature
