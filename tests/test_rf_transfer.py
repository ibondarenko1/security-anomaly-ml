import numpy as np

from tools.evaluate_rf_transfer import complete_lines, top_budget


def test_range_guard_keeps_only_complete_source_records():
    assert complete_lines(b"header\nrow-a\npartial", 0) == b"header\nrow-a\n"
    assert complete_lines(b"\nrow-b\npartial", 200) == b"row-b\n"
    assert complete_lines(b"tail\nrow-b\npartial", 201) == b"row-b\n"


def test_top_budget_orders_ties_by_original_row_without_looking_at_labels():
    np.testing.assert_array_equal(
        top_budget(np.array([1, 3, 3, 0]), 1), [False, True, False, False]
    )
