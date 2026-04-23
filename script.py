import pickle
from pathlib import Path
import numpy as np
import pandas as pd


df = pd.read_parquet('output/wesad_subject_stats_phase3.parquet')
subject_10 = df[df['subject'] == 'S10']
counts = subject_10['binary_label'].value_counts()
print(counts)