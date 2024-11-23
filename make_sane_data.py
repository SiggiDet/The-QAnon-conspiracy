import pandas as pd
import numpy as np
# Load the dataset into mem
df = pd.read_csv("./data/Users_isQ_words.csv")

# split into train/dev/test
np.random.seed(0)
indices = np.random.choice(a=[0, 1], size=len(df), p=[.999, .001])

print("Splitting data into train dev test")
sane = df[indices == 1]
sane.to_csv('./data/Users_isQ_words_sane.csv', index=False)

