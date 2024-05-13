import unittest
from src.model import FloodPrediction

class TestFloodPrediction(unittest.TestCase):

    def setUp(self):
        self.model = FloodPrediction(path='./data/')
        
        self.model.load_data("test_data.csv",
                        "test_target.csv",
                        data_index='Col1',
                        target_index='col1',
                        collapse_data_columns=['Col2','Col3','Col4'],
                        collapse_target_columns=['col2','col3','col4'],
                        data_column_name='new_data',
                        target_column_name='new_target',
                        drop_data_columns=['Col1'],
                        drop_target_columns=['col1']
        )

    def test_load_data(self):
        self.assertEqual(self.model.get_dataframe()['new_data'][0], 100.1, "Data loading failed.")
        self.assertEqual(self.model.get_dataframe()['new_target'][0], 0, "Target loading failed.")
        
    def test_make_data_target(self):   
        self.model.make_data_target()
        
        self.assertEqual(self.model.get_data()['new_data'][0], 100.1, "Data was not copied to X.")
        self.assertEqual(self.model.get_target()['new_target'][0], 0, "Target was not copied to y.")
        
    def test_apply_standard_scaler(self):
        self.model.make_data_target()
    
        self.model.apply_standard_scaler()
                
        self.assertAlmostEqual(self.model.get_data()['new_data'][0], 0, delta=2, msg="Data was not z-scored.")
        
    def test_apply_minmax_scaler(self):
        self.model.make_data_target()
    
        self.model.apply_minmax_scaler()
        
        self.assertAlmostEqual(self.model.get_data()['new_data'][0], 0.5, delta=0.5, msg="Data was not capped between 0 and 1.")

#    def test_apply_over_under_sampling(self):
#        self.model.make_data_target()
#        
#        self.model.apply_over_under_sampling()
#        
#        self.assertTrue(len(self.model.get_data()) > 2)

    def test_convert_to_supervised(self):
        self.model.make_data_target()
        
        self.model.convert_to_supervised()
        
        self.assertTrue(len(self.model.get_data().columns) > 1)
        
    def test_convert_shape_to_3d(self):
        self.model.make_data_target()
        
        self.model.convert_shape_to_3d(target_column='new_target')
        
        self.assertEqual(len(self.model.get_data().shape), 3, "Data not reshaped to 3d.")
        
    def test_train_test_split(self):
        self.model.make_data_target()
        
        self.model.train_test_split(test_size=0.5)
        
        self.assertAlmostEqual(len(self.model.get_data_test_split()), len(self.model.get_data())/2, delta=1, msg="Test data split not 50% of data.")

if __name__ == "__main__":
    unittest.main()
