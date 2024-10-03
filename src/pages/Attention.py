import streamlit as st
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from plots import plot_scatter
from page_attention.attention_lib import test_each_process


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


class AttentionNet(nn.Module):
    def __init__(self, D: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        
        self.Wk = nn.Linear(D, d_k)
        self.Wq = nn.Linear(D, d_k)
        self.Wv = nn.Linear(D, d_k)
        
        self.fc = nn.Linear(d_k, D)
            
    def forward(self, X): # X: batch_size x seq_len x D
        K = self.Wk(X) # K: batch_size x seq_len x d_k
        Q = self.Wq(X) # Q: batch_size x seq_len x d_k
        V = self.Wv(X) # V: batch_size x seq_len x d_k
        
        As = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        Aw = F.softmax(As, dim=-1)
        Ao = torch.matmul(Aw, V)
    
        Y = self.fc(Ao)
        return Y


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


def run():    
    anet = AttentionNet(D, K).to(device)

    Yt = test_each_process(anet, X_t, K)

    st.subheader('Out of the Attention Block: $Y_\mathrm{anet}[t] \equiv Y[t]$')
    Y = anet(X_t)
    display_with_plot(Y.cpu().detach(), 'Y')

    e = torch.abs(Y - Yt).sum()
    e = e.cpu().detach()
    st.write('Difference between $Y_\mathrm{anet}[t]$ and $Y[t]$: ')
    st.write(e)
    

run()