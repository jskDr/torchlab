import streamlit as st
import matplotlib.pyplot as plt
import torch

import plots

class KMeans:
    def __init__(self):
        pass 
    
    def data_preparation(self):
        st.header('Data Preparation')
        st.subheader('Input Parameters')
        cols = st.columns(4)
        N = cols[0].number_input('#amples: N', value=10)
        D = cols[1].number_input('Dimension: D', value=2)
        K = cols[2].number_input('#means: K', value=2)
        epsilon = cols[3].number_input('Epsilon: $\epsilon$', step=0.01, value=0.01)

        st.subheader('Random Input Data: $X \in R_{N \\times D}$')
        X = torch.rand(N, D)
        st.write(X)
        # if D == 2:
        #     st.write("Plot of $X$")
        #     fig, ax = plt.subplots()
        #     ax.scatter(X[:,0], X[:,1])
        #     # Add labels for each point
        #     for i, (x, y) in enumerate(zip(X[:,0], X[:,1])):
        #         ax.annotate(f'$x_{i}$({x:.2f}, {y:.2f})', (x, y), xytext=(5, 5), textcoords='offset points')
        #     st.pyplot(fig)
            
        self.N, self.D, self.K = N, D, K
        self.epsilon = epsilon
        self.X = X

    def computing_kmeans(self):
        N, D, K = self.N, self.D, self.K
        epsilon = self.epsilon
        X = self.X
        
        st.header('Computing KMeans')
        st.subheader('1. Initial Centroid: $C \in R_{K \\times D}$')
        C = X[:K,:]
        st.write(C)

        st.subheader('2. Iterative Computing')
        delta = float('inf')
        iter = 0
        while delta > epsilon:
            iter += 1
            st.write(f':blue[**Iteration: {iter}**]')
            st.write('A. Distance between sample and centroid: $L \in R_{N \\times K}$')
            L = []
            for k in range(K):
                dist = X - C[k]
                dist = torch.pow(dist, 2).sum(dim=1)
                L.append(dist)
            L = torch.stack(L,dim=1)
            st.write(L)

            st.write('B. Clustering data for each centroid: $g \in Z_{N}$')
            g = []
            for n in range(N):
                group = torch.argmin(L[n,:])
                g.append(int(group))
            st.write(g)

            st.write('C. Recalculate Centroid: $C \in R_{K \\times D}$')
            C_old = C
            C = torch.zeros(K,D)
            cnt = torch.zeros(K)
            for n, k in enumerate(g):
                cnt[k] += 1
                C[k,:] += X[n,:]
            for k, c in enumerate(cnt):
                C[k,:] /= c
            cols = st.columns(2)
            with cols[0]:
                st.write('$C$')
                st.write(C)
            with cols[1]:
                st.write('Old $C$')
                st.write(C_old)

            st.write('D. Compute Delta of C: $\delta$')
            delta = torch.pow(C - C_old,2).sum() / (K*D)
            cols = st.columns(2)
            with cols[0]:
                st.write('$\delta$')
                st.write(delta)
            with cols[1]:
                st.write('$C - C_\\textrm{old}$')
                st.write(C - C_old)
                
        self.C = C

    def kmeans_result(self):
        D = self.D
        X, C = self.X, self.C
        
        st.header('KMeans Result')
        st.subheader('Final KMeans: $C \in R_{K \\times D}$')
        st.write(C)
        
        if D == 2:
            st.write("Plot of $X$ and $C_o$")
            fig, ax = plt.subplots()
            plots.plot_scatter(X, fig, ax)
            for i, (x, y) in enumerate(zip(C[:,0], C[:,1])):
                ax.annotate(f'$c_{i}$({x:.2f}, {y:.2f})', (x, y), xytext=(-5, 5), textcoords='offset points')
            st.pyplot(fig)

def main():
    st.set_page_config(
        page_title='KMeans by PyTorch',
        layout='wide',
        page_icon=':fire:'
    )

    st.title("KMeans")

    kmeans = KMeans()
    kmeans.data_preparation()
    
    gcols = st.columns(2)
    gc0 = gcols[0].container(height=500) 
    gc1 = gcols[1].container(height=500)
    with gc0:
        kmeans.computing_kmeans()
    with gc1:
        kmeans.kmeans_result()
        
if __name__ == '__main__':
    main()