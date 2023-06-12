'''
Unit test template file for functions.py
'''
import unittest
import pandas as pd
import module_functions as module

class TestStringMethods(unittest.TestCase):
    '''
    Unit test for functions.py
    '''
    def test_import_data_check_na(self):
        '''Checks if there is any na in the data'''

        data_frame = module.import_data('data_/sales.csv', 'data_/store.csv')
        self.assertTrue(all(p == 0 for p in data_frame.isna().sum()))

    def test_import_data_check_size(self):
        '''Checks if the dataframe dimension is correct'''

        data_frame = module.import_data('data_/sales.csv', 'data_/store.csv')
        self.assertEqual(data_frame.shape[1], 21)

    def test_data_visualization1_graph_created(self):
        '''Checks if graph has a proper name'''
        mock_data = pd.DataFrame({'brand': ['A', 'B'], 'logmove': [3, 4], 'profit':[4,8]})
        mock_plt = module.data_visualization1(mock_data)

        self.assertEqual(mock_plt[1][0].get_title(), "Total Profit per Brand")
        self.assertEqual(mock_plt[1][1].get_title(), "Total Quantity Sold per Brand")

    def test_data_visualization2_graph_created(self):
        '''Checks if graph has a proper name'''
        mock_data = pd.DataFrame({'week': [1, 2], 'logmove': [3, 4], 'profit':[4,8]})
        mock_plt = module.data_visualization2(mock_data)

        self.assertEqual(mock_plt[1][0].get_title(), "Total Logmove over Week")
        self.assertEqual(mock_plt[1][1].get_title(), "Total Profit over Week")

    def test_descriptive_statistics_check_existent(self):
        '''Checks if function exists'''
        self.assertTrue(hasattr(module, 'descriptive_statistics'))

    def test_descriptive_statistics_check_highest_sales_dimension(self):
        '''Checks if the dimension returned of highest_sales is correct'''
        mock_data = module.import_data('data_/mock_sales.csv', 'data_/store.csv')
        result = module.descriptive_statistics(mock_data)[0]
        self.assertEqual(result.shape, (7,2))

    def test_descriptive_statistics_check_description_dimension(self):
        '''Checks if the dimension returned of highest_sales is correct'''
        mock_data = module.import_data('data_/mock_sales.csv', 'data_/store.csv')
        result = module.descriptive_statistics(mock_data)[1]
        self.assertEqual(result.shape, (8, 20))

    def test_models_check_existent(self):
        '''Checks the models function exists'''
        self.assertTrue(hasattr(module, 'models'))

    def test_models_ols_significance(self):
        '''Checks if the coefficient returned is interesting'''

        mock_data = module.import_data('data_/mock_sales.csv', 'data_/store.csv')

        result = module.models(mock_data)[0]
        coeff = result.params['price']

        self.assertNotEqual(coeff, 0)

if __name__ == '__main__':
    unittest.main()
