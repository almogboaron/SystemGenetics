import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def filter_neighboring_rows(data_frame, columns_to_check):
    # Create a boolean mask to identify rows where any of the specified columns differ from the previous row
    mask = ~data_frame[columns_to_check].eq(data_frame[columns_to_check].shift(1)).all(axis=1)
    
    # Apply the mask to filter the DataFrame
    filtered_df = data_frame[mask]
    
    return filtered_df




# Main Function for Hw3:

df_genotype = pd.read_excel('SystemGenetics\genotypes.xls' ,header = 1)
df_genotype.to_csv('SystemGenetics\genotypes.csv', index=False)
columns = df_genotype.columns.delete([0,1,2,3])
df_filtered_genotype= filter_neighboring_rows(df_genotype, columns)
print(df_filtered_genotype)
df_filtered_genotype.to_csv('SystemGenetics\genotypes_filtered.csv', index=False)