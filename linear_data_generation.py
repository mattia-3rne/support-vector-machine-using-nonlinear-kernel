from sklearn.datasets import make_blobs
import pandas as pd

# Generating 2D linearly separable data
X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=2.0)

# Converting to dataframe
df = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], label=y))

# Saving to CSV
filename = "data_linear.csv"
df.to_csv(filename, index=False)
print(f"Success: {filename} generated.")