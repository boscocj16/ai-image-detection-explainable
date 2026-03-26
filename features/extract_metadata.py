import os
import csv
from core.metadata import metadata_signal

dataset_real = "dataset/real"
dataset_ai = "dataset/ai"

output = "features/metadata.csv"

with open(output,"w",newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["filename","metadata","label"])

    for folder,label in [(dataset_real,0),(dataset_ai,1)]:

        for file in os.listdir(folder):

            path = os.path.join(folder,file)

            try:
                val = metadata_signal(path)
                writer.writerow([file,val,label])
            except:
                pass