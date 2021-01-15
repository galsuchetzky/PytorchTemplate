import unittest
import shutil
import os

TEST_DATA_DIR = './data/'

"""
TODO testing:
1) test downloading the dataset
2) test getting examples
3) test logger outputs
4) test operation modes of the dataset (train test eval)
5) test length
6) test lexicon creation
7) test iteration on the dataset
"""


class TestBREAKLogical(unittest.TestCase):
    def test_train_set(self):
        """
        Tests if configuring train returns the train dataset.
        """
        # self.dataset_logical_train = BREAKLogical('./', train=True, valid=False)
        # self.assertIsNotNone(self.dataset_logical_train, 'failed to load the dataset')

        if os.path.isdir('./data/'):
            shutil.rmtree('./data/')
        if os.path.isdir('./downloads/'):
            shutil.rmtree('./downloads/')
        if os.path.isdir('./break_data/'):
            shutil.rmtree('./break_data/')




if __name__ == '__main__':
    unittest.main()
