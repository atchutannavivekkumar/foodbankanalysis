import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Corrected file path with a raw string
data = pd.read_csv("/Users/vivekkumar/Downloads/Normal_weight__overweight__and_obesity_among_adults_aged_20_and_over__by_selected_characteristics__United_States.csv")

# Display the first 5 rows of the dataset
print(data.head())

#cleaning dataset 

Updated_data = data.drop('FLAG', axis=1)

#print(Updated_data.head()
newnull_counts = Updated_data.isnull().sum()
#print(newnull_counts)


mean_ESTIMATE = int(Updated_data['ESTIMATE'].mean())  # Calculate the mean and convert to an integer
print(mean_ESTIMATE)
Updated_data['ESTIMATE'] = Updated_data['ESTIMATE'].apply(lambda x: mean_ESTIMATE if pd.isnull(x) else x)  # Replace null values


mean_SE = int( Updated_data['SE'].mean())
#print(mean_house)
Updated_data['SE'] = Updated_data['SE'].apply(lambda x: mean_SE if pd.isnull(x) else x)


null_counts = Updated_data.isnull().sum()

print(null_counts)

#File path to save the updated dataset
updated_file_path = "/Users/vivekkumar/Desktop/updated_obesity.csv"

# Save the updated dataset as a CSV file
Updated_data.to_csv(updated_file_path, index=False)

#downloading dataset
#print("Updated dataset saved at: {updated_file_path}")