import ArcFace.model.prepare_data as prepare_data
import numpy as np


def test_preprocessing():
    testcase1 = np.random.rand(3, 5, 3)
    assert prepare_data.preprocessing(testcase1).shape == (3, 5, 3)
