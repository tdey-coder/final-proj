
"""
CSE 163 Final Project
Exploratory Data Analysis
Tina Dey
Autumn 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def read_data():
    """Read the exoplanet data from a CSV file."""
    return pd.read_csv("exoplanet_data.csv", comment="#",
                       usecols=["pl_name", "pl_rade", "pl_bmasse",
                                "pl_orbper", "discoverymethod",
                                "disc_year"])


def clean_data(data):
    """Remove rows with missing or invalid measurements."""
    required_columns = ["pl_name", "discoverymethod", "disc_year",
                        "pl_rade", "pl_bmasse", "pl_orbper"]

    clean = data.dropna(subset=required_columns)

    clean = clean[(clean["pl_rade"] > 0) & (clean["pl_bmasse"] > 0) &
                  (clean["pl_orbper"] > 0)]

    clean = clean.drop_duplicates(subset="pl_name")

    return clean


def select_sample(data, n=400):
    """Select up to n planets using a fixed random seed."""
    if len(data) > n:
        sample = data.groupby("discoverymethod", group_keys=False).sample(
                              frac=n / len(data), random_state=163)

        sample = sample.sample(n=n, random_state=163)
    else:
        sample = data.copy()

    return sample


def test_radius(sample_data):
    """Compare planet radius distributions by discovery method."""
    transit_radius = sample_data[
        sample_data["discoverymethod"] == "Transit"
    ]["pl_rade"]

    radial_radius = sample_data[
        sample_data["discoverymethod"] == "Radial Velocity"
    ]["pl_rade"]

    return stats.kruskal(transit_radius, radial_radius)


def test_period(sample_data):
    """Compare orbital period distributions by discovery method."""
    transit_period = sample_data[
        sample_data["discoverymethod"] == "Transit"
    ]["pl_orbper"]

    radial_period = sample_data[
        sample_data["discoverymethod"] == "Radial Velocity"
    ]["pl_orbper"]

    return stats.kruskal(transit_period, radial_period)


def test_mass_radius(sample_data):
    """Measure the relationship between planet mass and radius."""
    return stats.spearmanr(sample_data["pl_rade"],
                           sample_data["pl_bmasse"])


def main():
    """Load, clean, explore, and analyze the exoplanet dataset."""

    # Read the original dataset

    data = read_data()

    print("First five rows:")
    print(data.head())

    print()
    print("Dataset dimensions:")
    print(data.shape)

    print()
    print("Column names:")
    print(data.columns)

    # Examine missing values

    print()
    print("Missing values per column:")
    print(data.isna().sum())

    print()
    print("Percentage of missing values:")
    missing_percent = (data.isna().sum() / len(data)) * 100
    print(missing_percent)

    # Examine duplicates and invalid measurements

    print()
    print("Duplicate planet names:")
    print(data.duplicated(subset="pl_name").sum())

    print()
    print("Measurements that are zero or negative:")
    print((data[["pl_rade", "pl_bmasse", "pl_orbper"]] <= 0).sum())

    # Clean the dataset

    print()
    print("Cleaning the dataset:")

    cleaned_data = clean_data(data)

    print("Original number of planets:", len(data))
    print("Number of eligible planets:", len(cleaned_data))
    print(
        "Number of removed planets:",
        len(data) - len(cleaned_data)
    )

    # Select up to 400 planets

    print()
    print("Selecting up to 400 planets:")

    sample_data = select_sample(cleaned_data)

    print("Final sample size:", len(sample_data))

    # Summarize categorical variables

    print()
    print("Discovery method counts:")
    print(sample_data["discoverymethod"].value_counts())

    print()
    print("Unique discovery methods:")
    print(sample_data["discoverymethod"].unique())

    # Summarize quantitative variables

    print()
    print("Descriptive statistics:")

    numerical_columns = [
        "pl_rade",
        "pl_bmasse",
        "pl_orbper"
    ]

    print(sample_data[numerical_columns].describe())

    # Prepare data for discovery method comparisons

    comparison_data = sample_data[
        sample_data["discoverymethod"].isin(
            ["Transit", "Radial Velocity"])]

    # Check Kruskal-Wallis distribution assumptions

    print()
    print("Checking Kruskal-Wallis assumptions:")

    sns.histplot(data=comparison_data, x="pl_rade", hue="discoverymethod",
                 element="step", stat="density", common_norm=False)

    plt.title("Planet Radius Distributions by Discovery Method")
    plt.xlabel("Planet Radius (Earth Radii)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("radius_distribution_check.png")
    plt.show()
    plt.close()

    sns.histplot(data=comparison_data, x="pl_orbper", hue="discoverymethod",
                 element="step", stat="density", common_norm=False)

    plt.title("Orbital Period Distributions by Discovery Method")
    plt.xlabel("Orbital Period (Days)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig("period_distribution_check.png")
    plt.show()
    plt.close()

    # Visualization 1: Planet radius by discovery method

    print()
    print("Creating planet radius boxplot:")

    sns.boxplot(data=sample_data, x="pl_rade", y="discoverymethod")

    plt.title("Planet Radius by Discovery Method")
    plt.xlabel("Planet Radius (Earth Radii)")
    plt.ylabel("Discovery Method")
    plt.tight_layout()
    plt.savefig("radius_by_method.png")
    plt.show()
    plt.close()

    # Visualization 2: Planet mass vs. radius

    print()
    print("Creating planet mass vs. radius scatterplot:")

    sns.scatterplot(data=sample_data, x="pl_rade", y="pl_bmasse")

    plt.title("Planet Mass vs. Planet Radius")
    plt.xlabel("Planet Radius (Earth Radii)")
    plt.ylabel("Planet Mass (Earth Masses)")
    plt.tight_layout()
    plt.savefig("mass_vs_radius.png")
    plt.show()
    plt.close()

    # Visualization 3: Orbital period by discovery method

    print()
    print("Creating orbital period boxplot:")

    sns.boxplot(data=sample_data, x="pl_orbper", y="discoverymethod")

    plt.title("Orbital Period by Discovery Method")
    plt.xlabel("Orbital Period (Days)")
    plt.ylabel("Discovery Method")
    plt.tight_layout()
    plt.savefig("orbital_period_by_method.png")
    plt.show()
    plt.close()

    # Statistical analysis: Result Validity challenge goal

    # Research Question 1:
    # Does planet radius differ by discovery method?

    print()
    print("Kruskal-Wallis Test: Planet Radius by Discovery Method")

    radius_result = test_radius(sample_data)

    print("Test statistic:", radius_result.statistic)
    print("P-value:", radius_result.pvalue)

    if radius_result.pvalue < 0.05:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis.")

    # Research Question 2:
    # Is there a relationship between planet mass and radius?

    print()
    print("Spearman Correlation: Planet Mass vs. Planet Radius")

    correlation_result = test_mass_radius(sample_data)

    print("Spearman correlation:", correlation_result.statistic)
    print("P-value:", correlation_result.pvalue)

    if correlation_result.pvalue < 0.05:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis.")

    # Research Question 3:
    # Do orbital periods differ by discovery method?

    print()
    print("Kruskal-Wallis Test: Orbital Period by Discovery Method")

    period_result = test_period(sample_data)

    print("Test statistic:", period_result.statistic)
    print("P-value:", period_result.pvalue)

    if period_result.pvalue < 0.05:
        print("Reject the null hypothesis.")
    else:
        print("Fail to reject the null hypothesis.")


if __name__ == "__main__":
    main()
