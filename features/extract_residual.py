import os
import csv
from core.residual import residual_variance

dataset_real = "dataset/real"
dataset_ai = "dataset/ai"

output = "features/residual.csv"


with open(output,"w",newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["filename","residual","label"])

    for folder,label in [(dataset_real,0),(dataset_ai,1)]:

        for file in os.listdir(folder):

            path = os.path.join(folder,file)

            try:
                val = residual_variance(path)
                writer.writerow([file,val,label])
            except:
                pass