import pandas as pd

if __name__ == '__main__':
    eval_df = pd.read_csv("csv_files/training_data.csv")
    sample = eval_df.sample(n=25, random_state=0)
    sample.to_csv("csv_files/sample.csv", index=False)
