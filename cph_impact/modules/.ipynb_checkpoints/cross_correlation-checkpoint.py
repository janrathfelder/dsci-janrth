from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from numbers import Number
import scipy
from scipy import stats
import numpy as np
import pandas as pd

class CrossCorrelation(object):
    '''
    Calculates the cross-correlation either for the full
    data set or for areas if specified.
    '''
    
    def __init__(self):
        pass
    
    
    def cross_cor_mean_removal(self,
                              x1: List[float],
                              x2: List[float],
                              lag=0)-> np.ndarray:
        '''
        Calculates the 
        cross-correlation with
        mean removal of two
        time-series with option
        to specify length of lag:
        '''
        if type(x1) != np.ndarray:
            x1 = np.array(x1)
        if type(x2) != np.ndarray:
            x2 = np.array(x2)
            
        x1 = x1[~np.isnan(x1)]
        x2 = x2[~np.isnan(x2)]
        r_mean_remov = np.dot(x1[lag:-1]-np.mean(x1[lag:-1]).T,
                          x2[:-1-(lag)]-np.mean(x2[:-1-(lag)]))
        return r_mean_remov


    def cross_cor_mean_removal_weighted(self,
                                        x1: List[float],
                                        x2: List[float],
                                        lag=0)-> np.ndarray:
            '''
            Calculates the 
            cross-correlation with
            mean removal and 
            weighting by the sum of
            the std of two
            time-series with option
            to specify length of lag for x2:
            '''
            #cross_cor_mean_removal(x1,x2)
            if type(x1) != np.ndarray:
                x1 = np.array(x1)
            if type(x2) != np.ndarray:
                x2 = np.array(x2)

            x1 = x1[~np.isnan(x1)]
            x2 = x2[~np.isnan(x2)]
            n = len(x1)
            r_mean_remov = self.cross_cor_mean_removal(x1,x2,lag)
            std_product = np.sqrt(np.sum((x1[lag:-1]-np.mean(x1[lag:-1]))**2)) * \
                          np.sqrt(np.sum((x2[:-1-(lag)]-np.mean(x2[:-1-(lag)]))**2))
            r_weighted = r_mean_remov/std_product

            # Presumably, if abs(r) > 1, then it is only some small artifact of floating
            # point arithmetic.
            r = max(min(r_weighted, 1.0), -1.0)
            degrfr = n-2
            if abs(r) == 1.0:
                prob = 0.0
            else:
                t_squared = r*r * (degrfr / ((1.0 - r) * (1.0 + r)))
                prob = scipy.special.betainc(0.5*degrfr, 0.5, degrfr / (degrfr + t_squared))
            return r_weighted, prob 
        
        
    def area_cross_correlation(self,
                                df: pd.DataFrame,
                               sales: str ,
                               covariate: str,
                               number_of_lags: Number,
                               area=0) -> pd.DataFrame:
        '''
        Calculates the cross-correlation between two variables.
        Has the option to calculate the cross-correlation for
        specific areas (e.g.: brick) to account for area specific
        behaviors (and to prevent the possibility of Simpson paradox).

        : param df: pandas dataframe containing the variables
        : param sales: main target where we expect some effect of the covariate
        : param covariate: variable where we expect some effect onto sales
        : param number_of_lags: the number of  lags specifiied
        '''
        cors = []
        counter = []
        if area != 0:
            for location in df[area].unique():
                df_temp = df[df[area]==location]
                for i in np.arange(0,number_of_lags,1):
                    cors.append(self.cross_cor_mean_removal_weighted(df_temp[sales],
                                               df_temp[covariate],i)[0])
                    counter.append(i)

        else:
            for i in np.arange(0,number_of_lags,1):
                    cors.append(self.cross_cor_mean_removal_weighted(df[sales],
                                               df[covariate],i)[0])
                    counter.append(i)

        data = {'Lags': counter,'Correlation': cors}
        df_output = pd.DataFrame(data)
        return df_output

