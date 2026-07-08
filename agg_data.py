import pandas as pd


class AggData:
    def __init__(self):
        ...
        
    def agg_method(self, df: pd.DataFrame, delete_columns: list, pivot_table_args_dict: dict, no_relevant_cols_list: list):
        """
        Метод для агрегации данных и удаления нерелевантных столбцов
        df - ДатаФрейм для агрегации
        delete_columns - list с ненужными колонками
        pivot_table_arg_dict - dict где ключи - это index, columns, values, aggfunc для pivot_table
        no_relevant_cols - list с нерелевантными столбцами для удаления
        """
        def no_relevant_columns(df: pd.DataFrame, no_relevant_cols: list):
            # Удаление нерелевантных столбцов
            df_ = df.copy(deep=True)
            df_ = df_.drop(columns=no_relevant_cols)
            return df_

        agg_df = df.copy(deep=True)
        agg_df = agg_df.drop(delete_columns, axis=1)
        agg_df.set_index('client_id', inplace=True)

        # Преобразование таблицы к широкому виду с функцией агрегации sum по столбцу values
        pivot_agg_df = agg_df.pivot_table(index=pivot_table_args_dict['index'], 
                                          columns=pivot_table_args_dict['columns'],
                                          values=pivot_table_args_dict['values'],
                                          aggfunc=pivot_table_args_dict['aggfunc'])
        # Удаление двойного названия столбцов после pivot_table
        pivot_agg_df.columns = list(pivot_agg_df.columns.to_frame().macro_category.values)

        # Удаление нерелевантных столбцов
        pivot_agg_df = no_relevant_columns(df=pivot_agg_df, no_relevant_cols=no_relevant_cols_list)
        pivot_agg_df.fillna(0, inplace=True)
        return pivot_agg_df

