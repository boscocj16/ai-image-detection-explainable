import os
import csv
from core.ela import ela_score

dataset_real = "dataset/real"
dataset_ai = "dataset/ai"

output = "features/ela.csv"


with open(output, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["filename", "ela", "label"])

    for folder, label in [(dataset_real,0),(dataset_ai,1)]:

        for file in os.listdir(folder):

            path = os.path.join(folder,file)

            try:
                val = ela_score(path)
                writer.writerow([file,val,label])
            except:
                pass