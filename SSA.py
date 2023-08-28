import numpy as np
import pandas as pd
from scipy import linalg

class SSA(object):
    '''Singular Spectrum Analysis object'''
    def __init__(self, time_series):
        self.ts = pd.DataFrame(time_series)
        self.ts_name = self.ts.columns.tolist()[0]
        if self.ts_name == 0:
            self.ts_name = 'ts'
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
        self.freq = self.ts.index.inferred_freq

    @staticmethod
    def _printer(name, *args):
        '''Helper function to print messages neatly'''
        print('-'*40)
        print(name+':')
        for msg in args:
            print(msg)  

    @staticmethod
    def get_contributions(X=None, s=None):
        '''Calculate the relative contribution of each of the singular values'''
        lambdas = np.power(s, 2)
        frob_norm = np.linalg.norm(X)
        ret = pd.DataFrame(lambdas/(frob_norm**2), columns=['Contribution'])
        ret['Contribution'] = ret.Contribution.round(4)
        return ret[ret.Contribution > 0]

    @staticmethod
    def diagonal_averaging(hankel_matrix):
        '''Performs anti-diagonal averaging from the given hankel matrix
        Returns: Pandas DataFrame object containing the reconstructed series'''
        mat = np.matrix(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L, K), max(L, K)
        new = np.zeros((L, K))
        if L > K:
            mat = mat.T
        ret = []
        
        # Diagonal Averaging
        for k in range(1-K_star, L_star):
            mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
            mask_n = sum(sum(mask))
            ma = np.ma.masked_array(mat.A, mask=1-mask)
            ret.append(ma.sum()/mask_n)
        
        return pd.DataFrame(ret).rename(columns={0: 'Reconstruction'})

    def embed(self, embedding_dimension=None, suspected_frequency=None, verbose=False):
        '''Embed the time series with embedding_dimension window size.
        Optional: suspected_frequency changes embedding_dimension such that it is divisible by suspected frequency'''
        if not embedding_dimension:
            self.embedding_dimension = self.ts_N // 2
        else:
            self.embedding_dimension = embedding_dimension
        if suspected_frequency:
            self.suspected_frequency = suspected_frequency
            self.embedding_dimension = (self.embedding_dimension // self.suspected_frequency) * self.suspected_frequency

        self.K = self.ts_N - self.embedding_dimension + 1
        self.X = np.matrix(linalg.hankel(self.ts, np.zeros(self.embedding_dimension))).T[:, :self.K]
        self.X_df = pd.DataFrame(self.X)
        self.X_complete = self.X_df.dropna(axis=1)
        self.X_com = np.matrix(self.X_complete.values)
        self.X_missing = self.X_df.drop(self.X_complete.columns, axis=1)
        self.X_miss = np.matrix(self.X_missing.values)

    def decompose(self):
        '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace'''
        X = self.X_com
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = np.matrix(self.U), np.sqrt(self.s), np.matrix(self.V)
        self.d = np.linalg.matrix_rank(X)
        self.s_contributions = self.get_contributions(X, self.s)
        self.r = len(self.s_contributions[self.s_contributions > 0])
        self.r_characteristic = round((self.s[:self.r]**2).sum()/(self.s**2).sum(), 4)
        self.orthonormal_base = {i: self.U[:, i] for i in range(self.r)}

    def _forecast_prep(self, singular_values=None):
        '''Prepare for forecasting'''
        self.X_com_hat = np.zeros(self.X_complete.shape)
        self.verticality_coefficient = 0
        self.forecast_orthonormal_base = {}
        
        # Construct forecast orthonormal base
        if singular_values:
            for i in singular_values:
                try:
                    self.forecast_orthonormal_base[i] = self.orthonormal_base[i]
                except KeyError:
                    print(f"Key {i} not found in orthonormal_base. Skipping.")
        else:
            self.forecast_orthonormal_base = self.orthonormal_base

        self.R = np.zeros(self.forecast_orthonormal_base[0].shape)[:-1]
        for Pi in self.forecast_orthonormal_base.values():
            self.X_com_hat += Pi * Pi.T * self.X_com
            pi = np.ravel(Pi)[-1]
            self.verticality_coefficient += pi**2
            self.R += pi * Pi[:-1]
        self.R = np.matrix(self.R / (1 - self.verticality_coefficient))
        self.X_com_tilde = self.diagonal_averaging(self.X_com_hat)

    def forecast_recurrent(self, steps_ahead=12, singular_values=None):
        '''Forecast using recurrent methodology'''
        try:
            self.X_com_hat
        except AttributeError:
            self._forecast_prep(singular_values)
        
        # Forecasting logic
        self.ts_forecast = np.array(self.ts_v[0])
        for i in range(1, self.ts_N + steps_ahead):
            try:
                if np.isnan(self.ts_v[i]):
                    x = self.R.T * np.matrix(self.ts_forecast[max(0, i-self.R.shape[0]): i]).T
                    self.ts_forecast = np.append(self.ts_forecast, x[0])
                else:
                    self.ts_forecast = np.append(self.ts_forecast, self.ts_v[i])
            except IndexError:
                x = self.R.T * np.matrix(self.ts_forecast[i-self.R.shape[0]: i]).T
                self.ts_forecast = np.append(self.ts_forecast, x[0])
        self.forecast_N = i + 1
