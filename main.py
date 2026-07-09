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
    
    # 4. Получить df с топ-5 категорий по величине score
    top_5_category_agg_df: pd.DataFrame = agg_df[list_top_5_category].copy(deep=True)
    
    # 5. Получить df с новыми фичами 
    feature_eng_df = feature_engineering_object.new_feature(df=top_5_category_agg_df, name_col_income='income')

    # 6. Находим число кластеров
    clusterization_object.elbow_method(df=feature_eng_df)
    # Вывод по пункту 6: после числа k кластеров = 5 график метода Локтя перестаёт существенно уменьшаться

    # 7. Находим кластеры
    clusterization_object.search_cluster(df=feature_eng_df)

