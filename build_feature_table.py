import pandas as pd

meta = pd.read_csv("features/metadata.csv")
noise = pd.read_csv("features/noise.csv")
freq = pd.read_csv("features/frequency.csv")
residual = pd.read_csv("features/residual.csv")
ela = pd.read_csv("features/ela.csv")

df = meta.merge(noise,on=["filename","label"])
df = df.merge(freq,on=["filename","label"])
df = df.merge(residual,on=["filename","label"])
df = df.merge(ela,on=["filename","label"])

df.to_csv("features/features.csv",index=False)

print("Feature table created")