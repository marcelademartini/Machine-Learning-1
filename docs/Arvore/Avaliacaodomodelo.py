# -- coding: utf-8 --
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import StringIO
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from tabulate import tabulate
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1) Carregamento da base
# ============================================================
URL = "https://raw.githubusercontent.com/marcelademartini/Machine-Learning-1/refs/heads/main/Testing.csv"
df = pd.read_csv(URL)

# ============================================================
# 2) Pré-processamento
#    - remove 'id'
#    - codifica 'diagnosis' (B/M) -> 0/1
# ============================================================
if "id" in df.columns:
    df = df.drop(columns=["id"])

if "diagnosis" not in df.columns:
    raise