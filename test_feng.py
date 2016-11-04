import unittest
import feng
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler

pd_data = pd.DataFrame({'a': [1, 1, 1], 'b': [2, 2, 2]})
np_data = pd_data.values

class Identity(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return X


def get_identity_pipe(cols):
    return feng.pipeline.Pipeline([('a', Identity())], cols)

def get_identity_pipe_fu(cols):
    return feng.pipeline.FeatureUnion([
        ('a', feng.pipeline.Pipeline([
            ('a', Identity())
        ], cols))
    ])


identity_fu = feng.pipeline.FeatureUnion([
    ('a', Identity()),
    ('b', Identity())
], n_jobs=1)



class Tests(unittest.TestCase):


    def test_submodules_exported(self):
        submodules = ('importance', 'pipeline', 'preprocessing')
        members = dir(feng)
        self.assertTrue(all(module in members for module in submodules))

    def test_feature_union_with_pandas(self):
        transformed = identity_fu.transform(pd_data)
        self.assertEqual(list(transformed.columns), ['a_a', 'a_b', 'b_a', 'b_b'])

    def test_feature_union_with_numpy(self):
        transformed = identity_fu.transform(np_data)
        self.assertEqual(transformed.shape, (3, 4))

    def test_pipeline_with_pandas_select_col(self):
        cols = ['a']
        transformed = get_identity_pipe(cols).transform(pd_data)
        self.assertEqual(transformed.columns, cols)

    def test_pipeline_with_pandas_no_select(self):
        transformed = get_identity_pipe(None).transform(pd_data)
        self.assertEqual(list(transformed.columns), list(pd_data.columns))

    def test_pipeline_with_numpy_select_col(self):
        cols = [1]
        transformed = get_identity_pipe(cols).transform(np_data)
        self.assertTrue((transformed == np_data[:, cols]).all())

    def test_pipeline_with_numpy_no_select(self):
        transformed = get_identity_pipe(None).transform(np_data)
        self.assertTrue((transformed == np_data).all())

    def test_feature_union_with_pipeline_pd(self):
        cols = ['b']
        transformed = get_identity_pipe_fu(cols).transform(pd_data)
        self.assertEqual(list(transformed.columns), ['a_b'])

    def test_feature_union_with_pipeline_np(self):
        cols = [1]
        transformed = get_identity_pipe_fu(cols).transform(np_data)
        self.assertTrue((transformed == np_data[:, cols]).all())




if __name__ == '__main__':
    all_tests = unittest.TestLoader().loadTestsFromTestCase(Tests)
    unittest.TextTestRunner(verbosity=2).run(all_tests)
