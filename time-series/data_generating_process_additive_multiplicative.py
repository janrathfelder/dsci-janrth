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
                 ts_behaviour,
                 time_steps,
                 number_of_ts,
                 ar_y,
                 ar_x,
                 ar_x_to_y,
                 betas_y,
                 betas_x,
                 betas_x_to_y):
        self.ts_behaviour = ts_behaviour
        self.time_steps = time_steps
        self.number_of_ts = number_of_ts
        self.ar_y = ar_y
        self.ar_x = ar_x
        self.ar_x_to_y = ar_x_to_y
        self.betas_y = betas_y
        self.betas_x = betas_x
        self.betas_x_to_y = betas_x_to_y
        
    def ar_processes(self):
        # create AR ts:
        time = np.arange(self.time_steps)
        y_independent = []
        y_dependent = []
        covariate_x = []
        runs = []

        seasonality = list(np.random.normal(loc=5, scale=1, size=12)) * self.time_steps
        for n in range(0, self.number_of_ts):
            y_ind_temp = []
            y_dep_temp = []
            x = []

            # trend and noise for y
            trend_y = [t for t in time]
            residual_y = np.random.normal(loc=2, scale=.5, size=self.time_steps)

            # trend and noise for x
            trend_x = [t for t in time]
            residual_x = np.random.normal(loc=2, scale=.5, size=self.time_steps)

            for i in range(0, self.time_steps-(self.ar_x+1)):
                trend_temp_x = trend_x[0+i:self.ar_x+i]
                seasonality_temp_x = seasonality[0+i:self.ar_x+i]
                e_temp_x = residual_x[0+i:self.ar_x+i]
                
                trend_temp_x_same_t = trend_x[self.ar_x+i+1]
                seasonality_x_same_t = seasonality[self.ar_x+i+1]
                e_temp_x_same_t = residual_x[self.ar_x+i+1]
                
                if self.ts_behaviour == 'additive':
                    x_ts = [trend_temp_x[q]+seasonality_temp_x[q]+e_temp_x[q] 
                            for q in range(0,self.ar_x)]
                    sum_same_t_x = np.sum([trend_temp_x_same_t, 
                                     seasonality_x_same_t, 
                                     e_temp_x_same_t])
                else:
                    x_ts = [trend_temp_x[q]*seasonality_temp_x[q]*e_temp_x[q] 
                            for q in range(0,self.ar_x)]
                    sum_same_t_x = np.sum([trend_temp_x_same_t*
                                     seasonality_x_same_t*
                                     e_temp_x_same_t])
                x_ts.reverse()
                x.append(np.sum([np.sum(np.array(self.betas_x) * np.array(x_ts)),sum_same_t_x]))

            for i in range(0, self.time_steps-(self.ar_y+1)):
                trend_temp_y = trend_y[0+i:self.ar_y+i]
                seasonality_temp_y = seasonality[0+i:self.ar_y+i]
                e_temp_y = residual_y[0+i:self.ar_y+i]
                x_temp = x[0+i:i+self.ar_x_to_y]
                
                trend_temp_y_same_t = trend_y[self.ar_y+i+1]
                seasonality_temp_y_same_t = seasonality[self.ar_y+i+1]
                e_temp_y_same_t = residual_y[self.ar_y+i+1]

                if self.ts_behaviour == 'additive':
                    ar_ts = [trend_temp_y[q]+seasonality_temp_y[q]+e_temp_y[q] 
                             for q in range(0,self.ar_y)]
                    sum_same_t_y = np.sum([trend_temp_y_same_t, 
                                     seasonality_temp_y_same_t, 
                                     e_temp_y_same_t])
                    
                else:
                    ar_ts = [trend_temp_y[q]*seasonality_temp_y[q]*e_temp_y[q] 
                             for q in range(0,self.ar_y)]
                    sum_same_t_y = np.sum([trend_temp_y_same_t*
                                     seasonality_temp_y_same_t*
                                     e_temp_y_same_t])
                ar_ts.reverse()
                y_ind_temp.append(np.sum([sum_same_t_y,np.sum(np.array(self.betas_y) * np.array(ar_ts))]))
                y_dep_temp.append(np.sum([sum_same_t_y,np.sum(np.append(np.array(self.betas_y) * np.array(ar_ts),(np.array(self.betas_x_to_y) * np.array(x_temp))))]))
            y_independent.append(y_ind_temp)
            y_dependent.append(y_dep_temp)
            runs.append([n]*len(y_ind_temp))
            covariate_x.append(x)

        df = pd.DataFrame([np.concatenate(y_independent),
                   np.concatenate(y_dependent),
                   np.concatenate(covariate_x),
                   np.concatenate(runs)])
        df = df.T
        df = df.dropna()

        df.rename(columns={0:'y_rand', 1:'y_dep', 2:'x', 3:'brick_id'}, inplace=True)

        date_freq = [pd.date_range('2020-01-01', periods=len(y_ind_temp), 
                               freq='M')]*len(range(0,self.number_of_ts))
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