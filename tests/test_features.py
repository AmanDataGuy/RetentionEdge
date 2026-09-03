from retentionedge.data.load import filter_two_arm, load_hillstrom
from retentionedge.features.balance import check_randomization_balance
from retentionedge.features.preprocessing import FEATURE_COLUMNS, build_preprocessor


def _two_arm_df():
    return filter_two_arm(load_hillstrom(), treatment="womens")


def test_preprocessor_output_shape():
    df = _two_arm_df()
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df[FEATURE_COLUMNS])
    # one row per input row, no NaNs, purely numeric output
    assert transformed.shape[0] == len(df)
    assert not (transformed != transformed).any()  # NaN check without importing numpy


def test_balance_check_returns_one_row_per_feature():
    df = _two_arm_df()
    result = check_randomization_balance(df, treatment_col="treated", feature_cols=FEATURE_COLUMNS)
    assert set(result["feature"]) == set(FEATURE_COLUMNS)
    assert {"test", "statistic", "p_value", "smd", "flag"}.issubset(result.columns)
    assert result["p_value"].between(0, 1).all()


def test_randomization_actually_held():
    """Sanity check on the real dataset: since treatment was truly randomized,
    the large majority of features should NOT be flagged as imbalanced."""
    df = _two_arm_df()
    result = check_randomization_balance(df, treatment_col="treated", feature_cols=FEATURE_COLUMNS)
    assert result["flag"].sum() <= 1
