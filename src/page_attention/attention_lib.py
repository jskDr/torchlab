import streamlit as st
import matplotlib.pyplot as plt
import torch
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

def test_each_process(anet, X_t, K):
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

    return Yt