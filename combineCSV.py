import pandas as pd

# List of CSV file names
csv_files = ['AAPL_Historical_Data_1980-2000.csv', 'AAPL_Historical_Data_2000-2020.csv', 'AAPL_Historical_Data_2020-2024.csv']

# Read each CSV file and store in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('Combined_AAPL_Historical_Data.csv', index=False)