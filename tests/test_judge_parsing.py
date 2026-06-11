from src.judge import parse_rating


def test_clean_integer():
    r, ok = parse_rating("7")
    assert ok and abs(r - 6 / 9) < 1e-9


def test_min_and_max_map_to_unit_interval():
    assert parse_rating("1") == (0.0, True)
    assert parse_rating("10") == (1.0, True)


def test_embedded_in_text():
    r, ok = parse_rating("Rating: 9")
    assert ok and abs(r - 8 / 9) < 1e-9
    r2, ok2 = parse_rating("score is 8/10")
    assert ok2 and abs(r2 - 7 / 9) < 1e-9


def test_unparseable_is_zero_not_crash():
    assert parse_rating("no number here") == (0.0, False)
    assert parse_rating("") == (0.0, False)
    assert parse_rating("0") == (0.0, False)  # 0 is out of [1,10]
