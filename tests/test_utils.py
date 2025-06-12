from ogusa.utils import read_cbo_forecast
import numpy as np


def test_read_cbo_forecast():
    """
    Test that CBO data read as expected.
    """
    test_df = read_cbo_forecast()

    assert np.allclose(
        test_df.loc[test_df["year"] == 2026, "Y"].values[0], 24.2205
    )
