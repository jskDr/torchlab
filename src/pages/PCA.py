import streamlit as st
import torch
from sklearn import datasets

st.title('Principle Component Analysis (PCA)')

st.header('Load input data')
data = datasets.load_iris()
X = torch.tensor(data.data, dtype=torch.float32)
y = torch.tensor(data.target, dtype=torch.int64)
N, D = X.shape
st.write(f"N: {N}, D: {D}")

st.header('Processing')
st.subheader('Eigen Decompostion')
Rx = torch.matmul(X.T, X) / N # Eigen vectors
st.write('Eigen matrix')
st.write(Rx)

st.write('Eigen values and vectors')
eigenvalues, eigenvectors = torch.linalg.eigh(Rx)
st.write(eigenvalues)
st.write(eigenvectors)






