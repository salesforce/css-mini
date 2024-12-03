import pandas as pd
import pickle as pkl

# Read data from disk
all_data = pd.read_csv("usage_metrics.csv", index_col=0)
_product_data = pd.read_csv("product_metrics.csv", index_col=0)
_peer_dims_data = pd.read_csv("peer_dims.csv", index_col=0)

# Merge data
all_data[_product_data.columns] = _product_data
all_data[_peer_dims_data.columns] = _peer_dims_data

# Trim data so it runs faster
all_data = all_data.iloc[:1000]

# Get model weights
with open("css-model", "rb") as f:
    loaded_model = pkl.load(f)

# Batch predict
to_save_to_dlo = loaded_model.score(all_data)
print(to_save_to_dlo)