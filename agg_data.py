import pandas as pd


class AggData:
    def agg_method(df: pd.DataFrame, delete_columns: list, pivot_table_args_dict: dict):
        """
        Метод для агрегации данных 
        df - ДатаФрейм для агрегации
        delete_columns - list с ненужными колонками
        pivot_table_arg_dict - dict где ключи - это index, columns, values, aggfunc для pivot_table
        """
        
        agg_df = df.copy(deep=True)
        agg_df = agg_df.drop(delete_columns, axis=1)
        agg_df.set_index('client_id', inplace=True)
        pivot_agg_df = agg_df.pivot_table(index=pivot_table_args_dict['index'], 
                                          columns=pivot_table_args_dict['columns'],
                                          values=pivot_table_args_dict['values'],
                                          aggfunc=pivot_table_args_dict['aggfunc'])

        pivot_agg_df.columns = list(pivot_agg_df.columns.to_frame().macro_category.values)
        
        print('pivot_agg_df\n', pivot_agg_df)

