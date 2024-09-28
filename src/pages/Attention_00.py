import streamlit as st
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from plots import plot_scatter

def display_with_plot(X, symbol='x'):
    D = X.shape[1]
    if D == 2:
        cols = st.columns([2,1])
        cols[0].write(X)
        with cols[1]:
            fig, ax = plt.subplots()
            plot_scatter(X, fig, ax, symbol=symbol)
            st.pyplot(fig)
    else:
        st.write(X)


st.set_page_config(
        page_title='KMeans by PyTorch',
        layout='wide',
        page_icon=':fire:'
    )

st.title("Attention")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
st.caption(f"Device: {device}")

st.header('Data Preparation')
st.subheader('Input Parameters')
cols = st.columns(3)
N = cols[0].number_input('#Samples: N', value=3)
D = cols[1].number_input('Dimension: D', value=2)
K = cols[2].number_input('Attention: K', value=2)

st.subheader('Random Input Data: $X[t] \in R_{N \\times D}$')
X_t = torch.rand(N,D)
X_t = X_t.to(device)
display_with_plot(X_t.cpu(), 'X^t')
    
class AttentionNet(nn.Module):
    def __init__(self, D:int, d_k:int):
        super().__init__()
        self.d_k = d_k
        
        self.Wk = nn.Linear(D, d_k)
        self.Wq = nn.Linear(D, d_k)
        self.Wv = nn.Linear(D, d_k)
        
        self.fc = nn.Linear(d_k, D)
            
    def forward(self, X_t): # x: N x D
        K = self.Wk(X_t) # K: N x K, K = X_t Wk
        Q = self.Wq(X_t) # Q: N x K
        V = self.Wv(X_t) # V: N x K        
        
        As = torch.matmul(Q, K.transpose(0,1)) / (self.d_k ** 0.5)
        Aw = F.softmax(As, dim=1)
        Ao = torch.matmul(Aw, V)
    
        Y_t = self.fc(Ao)
        return Y_t
    
anet = AttentionNet(D,K).to(device)


st.header('Attention Process')
st.subheader('Key: $Y_k[t] = W_k X[t] \in R_{N \\times K}$')
Ykey = anet.Wk(X_t)
display_with_plot(Ykey.cpu().detach(), 'Y^k')

st.subheader('Query: $Y_q[t] = W_q X[t] \in R_{N \\times K}$')
Yquery = anet.Wq(X_t)
display_with_plot(Yquery.cpu().detach(), 'Y^q')

st.subheader('Value: $Y_v[t] = W_v X[t] \in R_{N \\times K}$')
Yvalue = anet.Wv(X_t)
display_with_plot(Yvalue.cpu().detach(), 'Y^v')

st.subheader('Attention scores: $As[t] = Y_q[t] Y_k[t]^T \in R_{N \\times N}$')
As_t = torch.matmul(Yquery, Ykey.transpose(0,1)) / (K**0.5)
display_with_plot(As_t.cpu().detach(), 'A^t')

st.subheader('Attention weights: $Aw[t] = \mathrm{softmax}(A[t]) \in R_{N \\times N}$')
Aw_t = F.softmax(As_t, dim=1)
display_with_plot(Aw_t.cpu().detach(), 'Aw^t')

st.subheader('Attention output: $A_o[t] = Aw[t] Y_v[t] \in R_{N \\times K}$')
Ao_t = torch.matmul(Aw_t, Yvalue) # N x K
display_with_plot(Ao_t.cpu().detach(), 'Ao^t')

st.subheader('Output of FC: $Y[t] \in R_{N \\times K}$')
st.write('FC layer recover the dimention into D from K again.')
Yt = anet.fc(Ao_t) # N x K
display_with_plot(Yt.cpu().detach(), 'Y^t')

st.subheader('Out of the Attention Block: $Y_\mathrm{anet}[t] \equiv Y[t]$')
Y = anet(X_t)
display_with_plot(Y.cpu().detach(), 'Y')

e = torch.abs(Y - Yt).sum()
e = e.cpu().detach()
st.write('Difference between $Y_\mathrm{anet}[t]$ and $Y[t]$: ')
st.write(e)