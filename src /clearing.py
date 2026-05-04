import pandas as pd

# 1. Define the filenames
main_file = "dataset_min_main.csv"
sub_file = "user_submissions.csv"

# 2. Load the actual data into DataFrames
df_main = pd.read_csv(main_file)
df_sub = pd.read_csv(sub_file)

# 3. Clear the data (keep the columns)
df_main = df_main.iloc[0:0]
df_sub = df_sub.iloc[0:0]

# 4. Save the now-empty files back to your folder
df_main.to_csv(main_file, index=False)
df_sub.to_csv(sub_file, index=False)

print("Files cleared successfully!")
