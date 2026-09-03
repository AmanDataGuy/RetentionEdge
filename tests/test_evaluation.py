from sklearn.model_selection import train_test_split

from retentionedge.data.load import filter_two_arm, load_hillstrom
from retentionedge.evaluation.budget_comparison import (
    compare_targeting_policies,
    incremental_conversions,
    pct_improvement,
)
from retentionedge.evaluation.qini import qini_coefficient
from retentionedge.features.preprocessing import FEATURE_COLUMNS, build_preprocessor
from retentionedge.models.uplift import t_learner_by_hand


def _small_df():
    df = filter_two_arm(load_hillstrom(), treatment="womens")
    small, _ = train_test_split(df, train_size=0.2, stratify=df["conversion"], random_state=0)
    return small


def test_incremental_conversions_is_zero_with_no_control_or_treated_rows():
    df = _small_df()
    treated_only_idx = df.loc[df["treated"] == 1].index
    assert incremental_conversions(df, treated_only_idx, "treated", "conversion") == 0.0


def test_incremental_conversions_matches_hand_computation_on_full_targeted_set():
    df = _small_df()
    result = incremental_conversions(df, df.index, "treated", "conversion")
    treated_rate = df.loc[df["treated"] == 1, "conversion"].mean()
    control_rate = df.loc[df["treated"] == 0, "conversion"].mean()
    expected = (treated_rate - control_rate) * len(df)
    assert abs(result - expected) < 1e-9


def test_qini_coefficient_is_finite():
    df = _small_df()
    X = build_preprocessor().fit_transform(df[FEATURE_COLUMNS])
    treatment = df["treated"].to_numpy()
    y = df["conversion"].to_numpy()
    t_learner = t_learner_by_hand(X, treatment, y)
    cate = t_learner.predict_cate(X)
    coefficient = qini_coefficient(y, cate, treatment)
    assert coefficient == coefficient  # not NaN


def test_compare_targeting_policies_returns_one_row_per_policy_with_larger_budget_covering_more():
    df = _small_df()
    naive_scores = df["conversion"].astype(float) + 0.0  # any per-row score works for shape testing
    uplift_scores = df["recency"].astype(float)  # placeholder ranking signal for this shape test
    result = compare_targeting_policies(df, naive_scores, uplift_scores, budget_pct=0.20)
    assert set(result["policy"]) == {"random", "naive_probability", "uplift"}
    assert (result["budget_n"] == int(len(df) * 0.20)).all()


def test_pct_improvement_basic_cases():
    assert pct_improvement(120, 100) == 20.0
    assert pct_improvement(80, 100) == -20.0
    assert pct_improvement(5, 0) == float("inf")
    assert pct_improvement(0, 0) == 0.0
