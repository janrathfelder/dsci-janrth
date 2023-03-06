import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

class stl_decomposition(object):
    '''
    Applies a STL decomposition using LOESS.
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
    '''
    
    def __init__(self,
                 grouping: str,
                 time_window: str):
        self.grouping = grouping
        self.time_window = time_window

    def decomposition(self,
                      df: pd.DataFrame,
                      target_value: str
                      ) -> pd.DataFrame:
        '''
        Decomposes a time-series into trend and
        seasonality and saves the trend, seasonality 
        and the residual.
        
            :param df: data frame for input
            :target_value: value to be decomposed into trend and seasonality
   
        '''
        seasonality = []
        trend = []
        detrend_deseaon = []

        for area in df[self.grouping].unique():
            temp = df[df[self.grouping]==area][[self.time_window, target_value]].copy()
            temp.index = pd.date_range(temp[self.time_window].iloc[0], 
                                      periods=len(temp), 
                                      freq='M')
            
            # use second highest value to override <=0 values
            temp[target_value] = np.where(temp[target_value]<=0,
                                                  sorted(set(temp[target_value].tolist()))[1], 
                                                  temp[target_value])

            result = STL(np.log(temp[target_value]), 
                         period=12,
                         seasonal_deg=0).fit()

            trend.append(np.exp(result.trend))
            seasonality.append(np.exp(result.seasonal))
            detrend_deseaon.append(np.exp(result.resid))

        df[target_value+'_trend_stl'] = np.concatenate(trend)
        df[target_value+'_season_stl'] = np.concatenate(seasonality)
        df[target_value+'_detrended_deseasonalized_stl'] = np.concatenate(detrend_deseaon)
        return df