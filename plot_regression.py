#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

## import packages and module ##


if __name__ == '__main__':
    dataframe = pd.read_csv("iris.csv")
    species = dataframe.species
    for species_name in species.unique():
        filtered_df = dataframe[dataframe['species'] == species_name]
        x = filtered_df.petal_length_cm
        y = filtered_df.sepal_length_cm
        regression = stats.linregress(x, y)
        slope = regression.slope
        intercept = regression.intercept
        plt.clf()
        plt.scatter(x, y, label = 'Data')
        plt.plot(x, slope*x + intercept, color = "orange", label = 'Fitted line')
        plt.xlabel("Petal length (cm)")
        plt.ylabel("Sepal length (cm)")
        plt.legend()
        plt.savefig(f"petal_v_sepal_length_regress_{species_name}.png")

        ##saves plots as .png files to working directory##
