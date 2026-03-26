import os
import csv
from core.frequency import high_frequency_ratio

dataset_real = "dataset/real"
dataset_ai = "dataset/ai"

output = "features/frequency.csv"

with open(output,"w",newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["filename","frequency","label"])

    for folder,label in [(dataset_real,0),(dataset_ai,1)]:

        for file in os.listdir(folder):

            path = os.path.join(folder,file)

            try:
                val = high_frequency_ratio(path)
                writer.writerow([file,val,label])
            except:
                pass