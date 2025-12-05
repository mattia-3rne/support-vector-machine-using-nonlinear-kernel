from sklearn.datasets import make_circles
import pandas as pd

# Generating 2D non-linear data
X, y = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=42)

# Converting to dataframe
df = pd.DataFrame(dict(x1=X[:,0], x2=X[:,1], label=y))

# Saving to CSV
filename = "data_nonlinear.csv"
df.to_csv(filename, index=False)
print(f"Success: {filename} generated.")