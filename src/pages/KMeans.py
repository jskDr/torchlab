import streamlit as st
import torch


st.title("KMeans")

st.header('Data Preparation')
st.subheader('Input Parameters')
cols = st.columns(3)
N = cols[0].number_input('Number of samples: N', value=3)
D = cols[1].number_input('Dimension of a sample vector: D', value=2)
K = cols[2].number_input('Number of the means: K', value=2)

st.subheader('Input Data: $X \in R_{N \\times D}$')
X = torch.rand(N, D)
st.write(X)

st.subheader('Initial Centroid: $C \in R_{K \\times D}$')
C = X[:K,:]
st.write(C)

st.subheader('Distance between each sample and each centroid: Dist')
for k in range(K):
    dist = X - C[k]
    dist = torch.pow(dist, 2).sum(dim=1)