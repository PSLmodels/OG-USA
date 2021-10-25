from ogusa import utils
import numpy as np


def test_read_cbo_forecast():
    """
    Test that CBO data read as expected.
    """
    test_df = utils.read_cbo_forecast()

    assert np.allclose(
        test_df.loc[test_df["year"] == 2017, "Y"].values[0], 20344
    )
