
"""
CSE 163 Final Project
Exploratory Data Analysis
Tina Dey
Autumn 2026
"""

import pandas as pd


def read_data():
    """Retrieve the exoplanet data from the NASA Exoplanet Archive."""
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+pl_name%2Cpl_rade%2Cpl_bmasse%2Cpl_orbper"
        "%2Cdiscoverymethod%2Cdisc_year+from+pscomppars"
        "&format=csv"
    )

    return pd.read_csv(url)


def main():
    """Load and inspect the NASA exoplanet dataset."""
    data = read_data()

    print("First five rows:")
    print(data.head())

    print("\nDataset dimensions:")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns)


if __name__ == "__main__":
    main()
