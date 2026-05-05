import pandas as pd

#Program to clear the user submission data
df_sub = pd.read_csv("user_submissions.csv")
df_sub = df_sub.iloc[0:0]
df_sub.to_csv(file, index=False)

print("The files have been cleared")
