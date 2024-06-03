import pandas as pd

# Load the dataset
file_path = 'top_10_counties_burned_data.csv'
df = pd.read_csv(file_path)

# Convert year_month to datetime for sorting
df['year_month'] = pd.to_datetime(df['year_month'])

# Sort by year_month
df_sorted = df.sort_values(by='year_month')

# Group by year_month and county, aggregating total acres and averaging the rest
df_aggregated = df_sorted.groupby(['year_month', 'county']).agg({
    'q_avgtempF': 'mean',
    'q_avghumid': 'mean',
    'GIS_ACRES': 'sum'
}).reset_index()

# Save the aggregated data
output_file_path = 'aggregated_top_10_counties_burned_data.csv'
df_aggregated.to_csv(output_file_path, index=False)
