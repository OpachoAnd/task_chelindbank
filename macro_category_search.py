import pandas as pd


class MacroCategorySearch:
    """
    Класс для поиска топ-5 кластеров 
    """
    def __init__(self):
        # Дисперсия и Активные клиенты
        self.stat_df = pd.DataFrame()
        # Нормализованные Дисперсия и Активные клиенты
        self.norm_stat_df = pd.DataFrame()

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
        Вернёт 2 DF: с нормализованными дисперсией и числом активных клиентов по макрокатегориям 
        и DF с ненормал. показателями

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
        stat_df = variance_df.merge(active_client_df, how='inner', on='category')
        
        # Нормализация дисперсии и числа активных клиентов
        norm_stat_df = stat_df.copy(deep=True)
        norm_stat_df[name_col_varience] = norm_stat_df[name_col_varience] / max(norm_stat_df[name_col_varience])
        norm_stat_df[name_col_active_client] = norm_stat_df[name_col_active_client] / max(norm_stat_df[name_col_active_client])

        return norm_stat_df, stat_df

    def get_top_five_category_df(self, 
                                 df: pd.DataFrame, 
                                 name_col_varience: str,
                                 name_col_category: str,
                                 name_col_active_client: str):
        """
        Метод вернет топ-5 макрокатегорий по величине score

        name_col_varience - Имя столбца для дисперсии
        name_col_category - Имя столбца Категория
        name_col_active_client - Имя столбца для числа активных клиентов
        """
        weight_var = 0.7  # Вес дисперсии для score
        weight_act_clients = 0.3  # Вес активных клиентов для score
        count_top_category = 5  # Количество категорий с максимальным score

        norm_stat_df, _ = self.get_norm_stat_df(df, name_col_varience, name_col_category, name_col_active_client)
        norm_stat_df['score'] = weight_var * norm_stat_df[name_col_varience] + weight_act_clients * norm_stat_df[name_col_active_client]

        # Выбор топ-5 категорий по score
        norm_stat_df.sort_values('score', ascending=False, inplace=True)
        norm_stat_df = norm_stat_df.head(count_top_category)

        return norm_stat_df[name_col_category].to_list()
