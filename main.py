import pandas as pd
from data_loader import DataLoader
from agg_data import AggData
from macro_category_search import MacroCategorySearch

PATH_FILE_DATA_TEST = 'datasets/data_test.xlsx'
PATH_FILE_MACRO_CATEGORIES = 'datasets/macro_categories.xlsx'

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    data_loader_object = DataLoader()
    agg_data_object = AggData()
    macro_category_object = MacroCategorySearch()

    # Загрузка необходимых датасетов и объединение их по полю category
    df = data_loader_object.data_load(path_data_test=PATH_FILE_DATA_TEST, 
                                      path_macro_categories=PATH_FILE_MACRO_CATEGORIES, 
                                      name_column_join='category', 
                                      how_join='inner')

    # Агрегация данных, удаление нерелевантных категорий, замена NaN на 0
    agg_df = agg_data_object.agg_method(df=df, 
                                        delete_columns=['category'], 
                                        pivot_table_args_dict={'index': 'client_id', 
                                                            'columns': 'macro_category',
                                                            'values': 'trans_amount',
                                                            'aggfunc': 'sum'},
                                        no_relevant_cols_list=['Финансы и платежи', 'Супермаркеты'])
    print('agg_df\n', agg_df, '\n')

    macro_category_object.variance_trans_amount(agg_df)


