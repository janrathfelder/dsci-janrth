import numpy as np
import seaborn as sns
import pandas as pd

class dgp_ts(object):
    '''
    Module for creating and exploring 
    multivariate time-series data with 
    auto-regressive nature for pooled
    regression analysis. 
    '''
    
    def __init__(self, 
                 n_sample,
                 periods,
                 ar_process_y,
                 ar_process_a,
                 rho_a1_mean,
                 rho_y1_mean,
                 rho_y2_mean,
                 rho_ya_mean,
                 rho_std_dev,
                 mean_trend_a,
                 mean_trend_y,
                 trend_std_dev):
        self.n_sample = n_sample
        self.periods = periods
        self.ar_process_y = ar_process_y
        self.ar_process_a = ar_process_a
        self.rho_a1_mean = rho_a1_mean
        self.rho_y1_mean = rho_y1_mean
        self.rho_y2_mean = rho_y2_mean
        self.rho_ya_mean = rho_ya_mean
        self.rho_std_dev = rho_std_dev
        self.mean_trend_a = mean_trend_a
        self.mean_trend_y = mean_trend_y
        self.trend_std_dev = trend_std_dev
        
    def ar_process(self
          ):
        '''
        Creates an artificial time-series with 
        length = periods and number of different
        samples = n_samples to perform pooled OLS
        (for example), where y is the target and
        a is a covariate.
        Rho (mean and std. dev.) defines the parameters 
        for the different autoregressive parts and the 
        slope of the trend can be changed by mean_trend.
        The output is a df, which contains one target 
        that is unrelated to a and one target that is
        an autoregressive process, which also depends 
        on a.
        '''
        rho_a1 = list(np.random.normal(self.rho_a1_mean,self.rho_std_dev,self.n_sample))
        rho_y1 = list(np.random.normal(self.rho_y1_mean,self.rho_std_dev,self.n_sample))
        rho_y2 = list(np.random.normal(self.rho_y2_mean,self.rho_std_dev,self.n_sample))
        rho_ya = list(np.random.normal(self.rho_ya_mean,self.rho_std_dev,self.n_sample))

        trend_a = list(np.random.normal(self.mean_trend_a,self.trend_std_dev,self.n_sample))
        trend_y = list(np.random.normal(self.mean_trend_y,self.trend_std_dev,self.n_sample))

        final_y_random = []
        final_y_dependent = []
        final_a = []
        runs = []

        for q in range(0,self.n_sample):
            y_rand = [np.random.uniform(0.2, .05), np.random.uniform(0.2, .05)]
            y_dep = [np.random.uniform(0.2, .05), np.random.uniform(0.2, .05)]
            a = [np.random.uniform(0.2, .05)]

            e_y = np.random.normal(.0,.1,self.periods)
            e_a = np.random.normal(.0,.2,self.periods)

            e_trend_a = np.random.normal(.0,.0,self.periods)
            e_trend_y = np.random.normal(.0,.0,self.periods)

            for i in range(len(e_a)):
                a.append(rho_a1[q]*a[i] + e_a[i] + (((i*trend_a[q])/2) + e_trend_a[i]))

            for i in range(len(e_y)):
                y_dep.append(rho_y1[q]*y_dep[i+1] + rho_y2[q]*y_dep[i] + \
                         e_y[i] + (((i*trend_y[q])/2) + e_trend_y[i]) + rho_ya[q]*a[i+1])

            for i in range(len(e_y)):
                y_rand.append(rho_y1[q]*y_rand[i+1] + rho_y2[q]*y_rand[i] + \
                         e_y[i] + (((i*trend_y[q])/2) + e_trend_y[i]))

            runs.append([q]*len(a))
            final_y_random.append(y_rand[:-1])
            final_y_dependent.append(y_dep[:-1])
            final_a.append(a)

        df = pd.DataFrame([np.concatenate(final_y_random),
                       np.concatenate(final_y_dependent),
                       np.concatenate(final_a),
                       np.concatenate(runs)])
        df = df.T
        df = df.dropna()

        df.rename(columns={0:'y_rand', 1:'y_dep', 2:'a', 3:'brick_id'}, inplace=True)

        date_freq = [pd.date_range('2020-01-01', periods=len(a), 
                                   freq='M')]*len(range(0,self.n_sample))
        df.index = np.concatenate(date_freq)
        return df
        
        
    def processing(self,
                   df,
                   cols, 
                   n_shift, 
                   n_rolling, 
                   rolling_method):
        '''
        Applys shifting and rolling window
        aggregation for specified columns.
        '''
        for col in cols:
            df[col+'_diff'] = df.groupby('brick_id')[col].diff()
        
        new_cols = list(df.columns)
        
        for col in new_cols:
            for t in range(1,n_shift+1):
                df[col+'_shift_'+str(t)] = df.groupby('brick_id')[col].shift(
                                                t)

            if rolling_method == 'mean':
                for ts in range(2,n_rolling+1):
                    df[col+'_'+str(ts)+str(rolling_method)] = df.groupby('brick_id'
                                                )[col].rolling(ts).mean().values
            else:
                for ts in range(2,n_rolling+1):
                    df[col+'_'+str(ts)+str(rolling_method)] = df.groupby('brick_id'
                                                )[col].rolling(ts).sum().values
        return df