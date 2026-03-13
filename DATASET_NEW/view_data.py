import pandas as pd

# path to dataset
file_path = "DATASET_NEW/Final_Dataset/maasarthi_master_dataset.csv"

# load dataset (only first 20 rows for speed)
df = pd.read_csv(file_path, nrows=20)

print("Dataset Preview:")
print(df)

print("\nColumns:")
print(df.columns)