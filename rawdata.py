import pandas as pd

# Path to your CSV file
file_path = r"C:\Users\KIIT\Desktop\bot docs\rawdataml.csv"

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Display the first few rows of the data
print(data.head())