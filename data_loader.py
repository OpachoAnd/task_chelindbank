import pandas as pd

class DataLoader:
    # Класс для получения данных
    def data_load(path_data_test: str, 
                  path_macro_categories: str, 
                  name_column_join: str, 
                  how_join: str = 'inner'):
        # Метод возвращающий данные в формате pd.DataFrame
        data_test_df = pd.read_excel(path_data_test)
        macro_categories = pd.read_excel(path_macro_categories)
        common_df = data_test_df.merge(macro_categories, on=name_column_join, how=how_join)
        return common_df
