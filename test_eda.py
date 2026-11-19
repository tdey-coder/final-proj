"""
Tests for the exoplanet EDA project.
Tina Dey
CSE 163 - Autumn 2026
"""

import pandas as pd
import eda


def make_test_data():
    """Create a small dataset for testing."""
    return pd.DataFrame({
        "pl_name": ["Planet A", "Planet B", "Planet C", "Planet D",
                    "Planet E", "Planet F", "Planet A"],
        "discoverymethod": ["Transit", "Transit", "Radial Velocity",
                            "Radial Velocity", "Transit",
                            "Radial Velocity", "Transit"],
        "disc_year": [2000, 2001, 2002, 2003, 2004, 2005, 2000],
        "pl_rade": [1, 2, 10, 11, 3, 12, 1],
        "pl_bmasse": [1, 4, 100, 121, None, 144, 1],
        "pl_orbper": [2, 3, 20, 30, 4, -5, 2]})


def test_clean_data():
    """Test that invalid and duplicate rows are removed."""
    data = make_test_data()

    cleaned = eda.clean_data(data)

    assert len(cleaned) == 4
    assert cleaned["pl_name"].is_unique
    assert cleaned.isna().sum().sum() == 0

    assert (cleaned["pl_rade"] > 0).all()
    assert (cleaned["pl_bmasse"] > 0).all()
    assert (cleaned["pl_orbper"] > 0).all()


def test_select_sample():
    """Test that sampling returns the correct number of rows."""
    data = eda.clean_data(make_test_data())

    sample = eda.select_sample(data, n=3)

    assert len(sample) == 3

    # Check that the same random seed gives the same sample.
    second_sample = eda.select_sample(data, n=3)

    assert sample.equals(second_sample)

    # Check that small datasets are not unnecessarily sampled.
    full_sample = eda.select_sample(data, n=10)

    assert len(full_sample) == len(data)


def test_statistical_tests():
    """Test that statistical functions return valid results."""
    data = eda.clean_data(make_test_data())

    radius_result = eda.test_radius(data)
    period_result = eda.test_period(data)
    correlation_result = eda.test_mass_radius(data)

    assert 0 <= radius_result.pvalue <= 1
    assert 0 <= period_result.pvalue <= 1
    assert 0 <= correlation_result.pvalue <= 1

    # The test data has a perfectly increasing mass-radius pattern.
    assert correlation_result.statistic == 1.0


def main():
    """Run all project tests."""
    test_clean_data()
    print("Cleaning tests passed!")

    test_select_sample()
    print("Sampling tests passed!")

    test_statistical_tests()
    print("Statistical tests passed!")

    print()
    print("All tests passed!")


if __name__ == "__main__":
    main()
