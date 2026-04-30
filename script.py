import pickle
from pathlib import Path
import numpy as np
import pandas as pd


df = pd.read_parquet('output/wesad_eda_features_phase3.parquet')
print(df.columns)