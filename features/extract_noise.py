import os
import csv
from core.noise import laplacian_variance

dataset_real = "dataset/real"
dataset_ai = "dataset/ai"

output = "features/noise.csv"

with open(output,"w",newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["filename","noise","label"])

    for folder,label in [(dataset_real,0),(dataset_ai,1)]:

        for file in os.listdir(folder):

            path = os.path.join(folder,file)

            try:
                val = laplacian_variance(path)
                writer.writerow([file,val,label])
            except:
                pass