from retentionedge.data.load import filter_two_arm, load_hillstrom

EXPECTED_COLUMNS = {
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
}


def test_load_hillstrom_shape_and_columns():
    df = load_hillstrom()
    assert len(df) == 64_000
    assert EXPECTED_COLUMNS.issubset(df.columns)
    assert set(df["segment"].unique()) == {"No E-Mail", "Mens E-Mail", "Womens E-Mail"}


def test_filter_two_arm_is_strict_subset():
    df = load_hillstrom()
    two_arm = filter_two_arm(df, treatment="womens")
    assert len(two_arm) < len(df)
    assert set(two_arm["segment"].unique()) == {"No E-Mail", "Womens E-Mail"}
    assert set(two_arm["treated"].unique()) == {0, 1}
    assert two_arm["treated"].sum() == (two_arm["segment"] == "Womens E-Mail").sum()


def test_filter_two_arm_rejects_unknown_treatment():
    df = load_hillstrom()
    try:
        filter_two_arm(df, treatment="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown treatment arm")
