import pandas as pd
from data_loader import DataLoader
from agg_data import AggData
from macro_category_search import MacroCategorySearch
from feature_engineering import FeatureEngineering
from clusterization import Clusterization


PATH_FILE_DATA_TEST = 'datasets/data_test.xlsx'
PATH_FILE_MACRO_CATEGORIES = 'datasets/macro_categories.xlsx'

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    data_loader_object = DataLoader()
    agg_data_object = AggData()
    macro_category_object = MacroCategorySearch()
    feature_engineering_object = FeatureEngineering()
    clusterization_object = Clusterization()

    # 1. Загрузка необходимых датасетов и объединение их по полю category
    df: pd.DataFrame = data_loader_object.data_load(path_data_test=PATH_FILE_DATA_TEST, 
                                      path_macro_categories=PATH_FILE_MACRO_CATEGORIES, 
                                      name_column_join='category', 
                                      how_join='inner')

    # 2. Агрегация данных, удаление нерелевантных категорий, замена NaN на 0
    agg_df: pd.DataFrame = agg_data_object.agg_method(df=df, 
                                        delete_columns=['category'], 
                                        pivot_table_args_dict={'index': 'client_id', 
                                                               'columns': 'macro_category',
                                                               'values': 'trans_amount',
                                                               'aggfunc': 'sum'},
                                        no_relevant_cols_list=['Финансы и платежи', 'Супермаркеты'])
    
    # 3. Получить список топ-5 категорий по величине score
    list_top_5_category: pd.DataFrame = macro_category_object.get_top_five_category_df(agg_df, 
                                        name_col_varience='variance',
                                        name_col_category='category', 
                                        name_col_active_client='active_client')
    # Получить расчитанные показатели по макрокатегориям
    conclusion_stat_df = macro_category_object.stat_df.copy(deep=True)
    conclusion_norm_stat_df = macro_category_object.norm_stat_df.copy(deep=True)
    conclusion_norm_stat_df.rename(columns={'variance': 'normed_variance', 'active_client': 'normed_active_client'}, 
                                   inplace=True)
    output_stat_df: pd.DataFrame = conclusion_stat_df.merge(conclusion_norm_stat_df, how='inner', on='category')

    # 4. Получить df с топ-5 категорий по величине score
    top_5_category_agg_df: pd.DataFrame = agg_df[list_top_5_category].copy(deep=True)
    
    # 5. Получить df с новыми фичами 
    feature_eng_df = feature_engineering_object.new_feature(df=top_5_category_agg_df, name_col_income='income')

    # 6. Находим число кластеров по методу локтя
    # График сохраняется в elbow_method_graph.png в корне проекта
    clusterization_object.elbow_method(df=feature_eng_df)

    # 7. Находим число кластеров по методу silhouette
    # График сохраняется в silhouette_method_graph.png' в корне проекта
    clusterization_object.silhouette_method(df=feature_eng_df)

    """
    Вывод по пунктам 6 и 7: 
    После числа k кластеров = 5 график метода Локтя перестаёт существенно уменьшаться
    При значении числа k кластеров = 5 на графике метода Silhouette самый высокий пик

    Следовательно, оптимальное число кластеров k = 5
    """

    # 8. Находим кластеры при числе кластеров k = 5
    k = 5
    # Клиенты с кластерами, Доля расходов кластеров в соответствующей макрокатегории
    clients_with_clusters_df, share_expenses_df = clusterization_object.search_cluster(df=feature_eng_df, k=k)


    """
    Сохранение выводов:
    """
    # Сохранение таблицы с расчитанными показателями по макрокатегориями
    output_stat_df.to_excel('output_stat_df.xlsx', index=False)

    # Сохранение таблицы Клиенты с кластерами
    clients_with_clusters_df.to_excel('clients_with_clusters_df.xlsx', index=True)

    # Сохранение таблицы Доля расходов кластеров в соответствующей макрокатегории
    share_expenses_df.to_excel('share_expenses_df.xlsx', index=True)
