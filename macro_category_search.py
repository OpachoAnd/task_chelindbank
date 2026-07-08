import pandas as pd


class MacroCategorySearch:
    """
    Класс для поиска топ-5 кластеров 
    """
    def __init__(self):
        self.stat_df = pd.DataFrame()

    def get_variance_df(self, 
                        df: pd.DataFrame, 
                        name_col_category: str,
                        name_col_varience: str):
        """
        Вернёт DF с дисперсией
        """
        variance_df = pd.DataFrame(df.var(), columns=[name_col_varience])
        variance_df.reset_index(names=name_col_category, inplace=True)
        return variance_df

    def get_active_client_df(self, 
                             df: pd.DataFrame,
                             name_col_category: str,
                             name_col_count_active_client: str):
        """
        Вернёт DF с количеством активных клиентов
        """
        mask = df > 0
        count_dict = {col: int(mask[col].sum(axis=0)) for col in df.columns}
        active_client_df = pd.DataFrame(list(count_dict.items()), 
                                        columns=[name_col_category, name_col_count_active_client])
        return active_client_df
    
    def get_norm_stat_df(self, 
                         df: pd.DataFrame, 
                         name_col_varience: str,
                         name_col_category: str,
                         name_col_active_client: str):
        """
        Вернёт DF с нормализованными дисперсией и числом активных клиентов по макрокатегориям
        name_col_varience - Имя столбца для дисперсии
        name_col_category - Имя столбца Категория
        name_col_active_client - Имя столбца для числа активных клиентов
        """
        variance_df: pd.DataFrame = self.get_variance_df(df, 
                                                         name_col_category, 
                                                         name_col_varience)
        active_client_df: pd.DataFrame = self.get_active_client_df(df, 
                                                                   name_col_category, 
                                                                   name_col_active_client)
        # Соединение DF дисперсии и акт. клиентов
        norm_stat_df = variance_df.merge(active_client_df, how='inner', on='category')
        print('norm_stat_df\n', norm_stat_df)
        
        # Нормализация дисперсии и числа активных клиентов
        norm_stat_df[name_col_varience] = norm_stat_df[name_col_varience] / max(norm_stat_df[name_col_varience])
        norm_stat_df[name_col_active_client] = norm_stat_df[name_col_active_client] / max(norm_stat_df[name_col_active_client])
        print('variance_df\n', variance_df)
        print('active_client_df\n', active_client_df)
        print('norm_stat_df\n', norm_stat_df)
