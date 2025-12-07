# Support Vector Machine using non-linear Kernel

## 📊 Project Overview

The goal of this project is to demonstrate the mathematical foundation and practical application of the Kernel Trick in Support Vector Machines (SVM).

This project implements two distinct classification scenarios to prove that while linear models are sufficient for simple datasets, complex geometries (like concentric circles) require mapping data into higher-dimensional feature spaces. By using the Radial Basis Function (RBF) kernel, we can construct decision boundaries that are non-linear in the original space but linear in the high-dimensional projection.

---

## 🧠 Theoretical Background

### Support Vector Machines

The Support Vector Machine is a supervised learning algorithm. In the general case of two features, the goal is to find a **line** that separates the two classes by a gap that is as wide as possible. This same logic applies to higher-dimensional data, where the separator becomes a hyperplane.

Given a set of training vectors $\mathbf{x}_i \in \mathbb{R}^p$ and their corresponding target values $y_i \in \{-1, 1\}$, the SVM seeks to find the weight vector $\mathbf{w}$ and bias $b$ that satisfy the following hyperplane equation:

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

To find the optimal separation, we minimize the squared norm of the weight vector, which maximizes the geometric margin $\frac{2}{||\mathbf{w}||}$:

$$
\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2
$$

Subject to the constraints that classify each training point correctly:

$$
y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
$$

---

### The Kernel Trick

For data that is not linearly separable in its original space (e.g., $\mathbb{R}^2$), strictly linear constraints generally result in no solution or high classification error. To solve this, we map the input vectors $\mathbf{x}$ into a higher-dimensional feature space $\mathcal{H}$ via a function $\phi(\mathbf{x})$.

The optimization problem in the dual form depends only on the dot products of the data points. This allows us to apply the Kernel Trick: we replace the dot product in the high-dimensional space with a kernel function $K$ computed in the original space:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)
$$

This avoids the computationally expensive (and sometimes impossible) step of explicitly calculating the coordinates in the infinite-dimensional space $\mathcal{H}$.

#### The Gram Matrix
The central mathematical component enabling this is the **Gram Matrix**, denoted as $G$. This is an $N \times N$ symmetric matrix containing all pairwise kernel evaluations. The diagonal elements represent the similarity of a point with itself:

$$
G =
\begin{pmatrix}
K(\mathbf{x}_1, \mathbf{x}_1) & K(\mathbf{x}_1, \mathbf{x}_2) & \cdots & K(\mathbf{x}_1, \mathbf{x}_N) \\
K(\mathbf{x}_2, \mathbf{x}_1) & K(\mathbf{x}_2, \mathbf{x}_2) & \cdots & K(\mathbf{x}_2, \mathbf{x}_N) \\
\vdots & \vdots & \ddots & \vdots \\
K(\mathbf{x}_N, \mathbf{x}_1) & K(\mathbf{x}_N, \mathbf{x}_2) & \cdots & K(\mathbf{x}_N, \mathbf{x}_N)
\end{pmatrix}
$$

The SVM dual optimization problem is rewritten entirely in terms of this matrix. We seek to maximize the Lagrangian dual function involving Lagrange multipliers $\alpha$:

$$
\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j G_{ij}
$$

Subject to:

$$
\sum_{i=1}^N \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C
$$

Because the optimization depends solely on $G_{ij}$, the complexity of training is determined by the number of samples $N$, rather than the dimensionality of the feature space.

---

### Radial Basis Function Kernel

This project utilizes the **Gaussian RBF Kernel**, which defines similarity based on the Euclidean distance between points. It creates a decision boundary by placing a Gaussian distribution over each support vector.

Mathematically, the RBF kernel is defined as:

$$
K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma ||\mathbf{x} - \mathbf{x}'||^2\right)
$$

Where:
* $||\mathbf{x} - \mathbf{x}'||^2$ is the squared Euclidean distance.
* $\gamma$ is a free parameter that defines the influence of a single training example. High $\gamma$ implies a close reach and thus high complexity, while low $\gamma$ implies a far reach and smoother decision boundary.

## ⚙️ Methodology

This project does not perform a brute-force comparison of all kernels on all datasets. Instead, it demonstrates the appropriate application of specific kernels based on the known geometry of the data.

### 1. The linear Case
* **Dataset**: Two distinct, linearly separable blobs.
* **Kernel**: `linear`
* **Mathematical Basis**: $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$
* **Result**: The SVM successfully builds a straight line boundary $\mathbf{w}^T \mathbf{x} + b = 0$.

### 2. The non-linear Case
* **Dataset**: Concentric circles.
* **Kernel**: `rbf`
* **Mathematical Basis**: Infinite-dimensional projection via Gaussians.
* **Result**: The SVM builds a circular decision boundary that wraps around the inner class, which corresponds to a linear hyperplane in the projected space.

---

## ⚠️ Limitations

| Limitation | Description |
| :--- |:---|
| **A Priori Knowledge** | This implementation assumes the user knows the geometry of the data beforehand. It does not automatically select the kernel; the choice is hardcoded to the specific problem type. |
| **Computational Complexity** | While the Kernel Trick avoids calculating $\phi(\mathbf{x})$, computing the Gram matrix $K(\mathbf{x}_i, \mathbf{x}_j)$ still scales quadratically with the number of samples $O(n^2)$, making it slow for very large datasets. |
| **Hyperparameter Sensitivity** | The RBF kernel requires careful tuning of $C$ for regularization and $\gamma$. Incorrect values can easily lead to overfitting or underfitting. |

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/support-vector-machine-using-non-linear-kernel.git
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis**:
    This project is structured into two independent notebooks. Each notebook handles its own data generation and model training.
    
    * Open `svm_linear.ipynb` to see the standard linear SVM.
    * Open `svm_non_linear.ipynb` to see the non-linear kernel implementation.

    ```bash
    jupyter notebook
    ```

---

## 📂 Project Structure

* `linear_data_generation.py`: Python script for linear data generation.
* `non_linear_data_generation.py`: Python script for non-linear data generation.
* `svm_linear.ipynb`: Notebook implementation of the linear case.
* `svm_non_linear.ipynb`: Notebook implementation of the non-linear case.
* `requirements.txt`: Python package dependencies.
* `README.md`: Project documentation.