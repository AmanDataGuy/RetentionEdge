from sklearn.model_selection import train_test_split

from retentionedge.data.load import filter_two_arm, load_hillstrom
from retentionedge.features.preprocessing import FEATURE_COLUMNS, build_preprocessor
from retentionedge.models.baseline import (
    evaluate_classifier,
    fit_logistic_regression,
    fit_random_forest,
    fit_xgboost,
    split_data,
)
from retentionedge.models.uplift import (
    segment_uplift_quadrants,
    t_learner_by_hand,
    t_learner_sklift,
    x_learner_by_hand,
)

METRIC_KEYS = {"threshold", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}


def _small_df():
    # Subsample for test speed: stratified sample keeps some positives in a small slice.
    df = filter_two_arm(load_hillstrom(), treatment="womens")
    small, _ = train_test_split(df, train_size=0.15, stratify=df["conversion"], random_state=0)
    return small


def test_split_data_is_stratified_and_disjoint():
    df = _small_df()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    # positive-class rate should be roughly preserved across splits (stratified)
    base_rate = df["conversion"].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - base_rate) < 0.03


def test_fit_logistic_regression_returns_fitted_pipeline():
    df = _small_df()
    X_train, _, _, y_train, _, _ = split_data(df)
    model = fit_logistic_regression(X_train, y_train)
    assert hasattr(model, "predict_proba")
    proba = model.predict_proba(X_train)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fit_random_forest_and_xgboost_return_fitted_pipelines():
    df = _small_df()
    X_train, _, _, y_train, _, _ = split_data(df)
    rf = fit_random_forest(X_train, y_train, n_iter=2, cv=2)
    xgb = fit_xgboost(X_train, y_train, n_iter=2, cv=2)
    for model in (rf, xgb):
        assert hasattr(model, "predict_proba")


def test_evaluate_classifier_returns_all_metric_keys():
    df = _small_df()
    X_train, _, X_test, y_train, _, y_test = split_data(df)
    model = fit_logistic_regression(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)
    assert METRIC_KEYS.issubset(metrics.keys())
    assert 0 <= metrics["roc_auc"] <= 1


def _preprocessed_uplift_inputs():
    df = _small_df()
    X = build_preprocessor().fit_transform(df[FEATURE_COLUMNS])
    treatment = df["treated"].to_numpy()
    y = df["conversion"].to_numpy()
    return X, treatment, y


def test_t_learner_by_hand_and_sklift_agree_on_shape_and_range():
    X, treatment, y = _preprocessed_uplift_inputs()
    hand = t_learner_by_hand(X, treatment, y)
    cate_hand = hand.predict_cate(X)
    assert cate_hand.shape == (len(X),)
    assert ((cate_hand >= -1) & (cate_hand <= 1)).all()

    sklift_model = t_learner_sklift(X, treatment, y)
    cate_sklift = sklift_model.predict(X)
    assert cate_sklift.shape == (len(X),)


def test_x_learner_by_hand_returns_finite_cate():
    X, treatment, y = _preprocessed_uplift_inputs()
    x_learner = x_learner_by_hand(X, treatment, y)
    cate = x_learner.predict_cate(X)
    assert cate.shape == (len(X),)
    assert (cate == cate).all()  # no NaNs


def test_segment_uplift_quadrants_covers_all_rows_with_known_labels():
    X, treatment, y = _preprocessed_uplift_inputs()
    t_learner = t_learner_by_hand(X, treatment, y)
    cate = t_learner.predict_cate(X)
    baseline_proba = t_learner.model_control.predict_proba(X)[:, 1]

    quadrants = segment_uplift_quadrants(cate, baseline_proba)
    assert len(quadrants) == len(X)
    assert set(quadrants.unique()).issubset({"persuadable", "sleeping_dog", "sure_thing", "lost_cause"})
