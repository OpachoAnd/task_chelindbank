import pandas as pd


class AggData:
    def agg_method(df: pd.DataFrame, delete_columns: list):
        # Метод для агрегации данных 
        # df - ДатаФрейм для агрегации
        # delete_columns - list с ненужными колонками
        agg_df = df.copy(deep=True)
        agg_df = agg_df.drop(delete_columns, axis=1)
        pivot_agg_df = agg_df.pivot_table(index='client_id', 
                                    columns='macro_category',
                                    values='trans_amount',
                                    aggfunc='sum')
        pivot_agg_df.reset_index(inplace=True)
        # pivot_agg_df.drop('macro_category', axis=1, inplace=True)
        print('pivot_agg_df\n', pivot_agg_df)

