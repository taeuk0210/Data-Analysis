
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

id_map = pd.read_csv("/home/aiuser/taeuk/open-problems-single-cell-perturbations/id_map.csv")
submit = pd.read_csv("/home/aiuser/taeuk/open-problems-single-cell-perturbations/sample_submission.csv")
adata_obs_meta = pd.read_csv("/home/aiuser/taeuk/open-problems-single-cell-perturbations/adata_obs_meta.csv")
adata_train = pd.read_parquet("/home/aiuser/taeuk/open-problems-single-cell-perturbations/multiome_train.parquet", engine="pyarrow")
id_map.head()

# 30초
gene = sorted(submit.columns[1:])
location = sorted(adata_train.location.unique())
print("# genes :",len(gene))
print("# genes in adata_train :",len(location))

adata_train["is_gene"] = pd.Series([False]*adata_train.shape[0])
adata_train.head()

for i in tqdm(range(adata_train.shape[0])):
    if adata_train['location'][i] in gene:
        adata_train["is_gene"][i] = True
        
df = adata_train.loc[adata_train["is_gene"] == True, :]
df.to_csv("./adata_split.csv", header=True, index=False)