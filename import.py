from datasets import load_dataset

ds = load_dataset("giseldo/neodataset")

df = ds["train"].to_pandas()

print(df.head())




