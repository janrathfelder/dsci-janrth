import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import scipy.stats
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.compat import lzip
from statsmodels.stats.stattools import durbin_watson
import matplotlib 

from typing import List
#from io import IOBytestream


class ols_check(object):
    '''
    Checks all relevant assumptions of OLS
    for time-series data.
    '''
    
    def __init__(self,
                 y_true: pd.Series,
                 y_pred: pd.Series,
                 features: List[str],
                 ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.features = features
        self.residuals = y_true - y_pred
        
    def __cooks_dist_line(self,
                          factor: float,
                          #features: List[str],
                          leverage: pd.Series) -> pd.Series:
        """
        Helper function for plotting Cook's distance curves
        """
        n = len(self.features)
        formula = lambda x: np.sqrt((factor * n * (1 - x)) / x)
        x = np.linspace(0.001, max(leverage), 50)
        y = formula(x)
        return x, y
    
    def __cooks_distance(self,
                         leverage: pd.Series,
                         residuals: pd.Series,
                         fittedvalues: pd.Series,
                         y_true: pd.Series) -> pd.Series:
        '''
        Helper function to calculate Cook's distance,
        which is given by: math::
            D_{i} = res_{i}**2/(p*s**2)[leverage_{i}/(1-leverage_{ii})**2],
        where:
            res = residual
            p = number of features
            s**2 = np.dot(res.T,res)/(n-p), with n = number of observations
            
        The input params are defined as:
            :param leverage: measure of how far away the independent variable values of an observations are from those of the other observations
            :param features: features used in the model equation
            :param residuals: the difference between the true values and the fitted values
            :param fittedvalues: the predictions
            :param y_true: the true target values
        '''
        n = len(residuals)
        p = len(features)
        mse = np.dot(residuals, residuals) / (n-p)
        cooks_distance = ((residuals)**2 / (p*mse)) * (leverage/(1-leverage)**2)
        return cooks_distance
    
    def __residuals_studentized_internal(self,
                                         df: pd.DataFrame,
                                         residuals: pd.Series,
                                         features: List[str]) -> pd.Series:
        '''
        Calculates internal studentized residuals.
        Also produces the leverage (h_{ii}), which
        will not only be used here but also for 
        Cook's distance for example.
        It is given by: math::
            t_{i} = res_{i} / \sigma * sqrt(1-h_{ii}),
        where: math::
            sigma**2 = 1/(n-p) sum_{i=1}^n res_{i}**2, 
            with: n = number of observations and
                  p = number of features
        h_{ii} is the diagonal of the Hat Matrix (H),
        which is given by: math::
            H = X(X.T*X)^{-1}X.T
            h = diag(H) 
        Note that in order to avoid a singular Matrix
        issue (e.g.: many dummy variables used) a generalized
        inverse technique is used to calculate the hat matrix.
        This might lead to small deviations in situations in
        which a regual inverse might be possible, but these
        deviations are considered to be really small.
        
        The input params are defined as follows:
            :param df: the input dataframe
            :param residuals: the difference between the true values and the fitted values
            :param features: features used in the model equation
        '''
        X = df[features].values

        n = len(residuals) # number of data points 
        p = len(features) # number of features

        sigma_res = (1/(n-p))*np.sum((residuals**2)) # variance of residuals

        hat_matrix = X.dot(np.linalg.pinv(X.T.dot(X)).dot(X.T)) # hat matrix
        leverage = np.diagonal(hat_matrix) # hat matrix diagonal
        res_studentized_internal = np.array(residuals / (np.sqrt(sigma_res* (1-leverage))))
        return leverage, res_studentized_internal 
    
    def vif_test(self, 
                 df_vif: pd.DataFrame) -> pd.DataFrame:
        '''
        Calculates the variance inflation factor
        for between all columns in the specified df.
        Outputs a filtered df with values>=5, which is an
        indication of strong multicollinearity.
        
        The input params are defined as follows:
            :param df_vif: the input dataframe, which contains all features used in the regression
        '''
        df_vif = df_vif.dropna()
        df_vif = add_constant(df_vif)
        vif = pd.DataFrame()
        vif["vif"] = pd.Series([variance_inflation_factor(df_vif.values, i) 
                   for i in range(df_vif.shape[1])], 
                  index=df_vif.columns)
        #vif_final = vif[(vif.vif>=5)].T.drop('const', axis=1).copy()
        return vif.sort_values('vif')
    
    def histogram_residuals(self,
                           residuals: pd.Series,
                           ax=None) -> matplotlib.axis.Axis:
    #IOBytestream:
        '''
        Plots a histogram of residuals with
        +/- 1 and 2 standard deviations
        
        The input params are defined as follows:
            :param residuals: residuals as the difference between true and prediction
        '''
        if ax is None:
            fig, ax = plt.subplots()

        res_mean = round(np.mean(residuals),2)
        res_std = round(np.std(residuals),2)
        sns.histplot(residuals, ax=ax)
        ax.set_title('Residuals', fontweight="bold")
        return ax
    
    def homoscedasticity_test_breusch_pagan(self,
                                            df,
                                            features):
        '''
        Test for homoscedasticity in the residuals (u).
        H0: Homoscedasticity is present
        If pvalue is small, we reject H0 and
        conclude that we have a heteroscedasticity
        problem in our data:
        .. math::
            u^2 = a_{0} + a_{1}*x_{1} + \cdots + a_{k}*x_{k} + v
            H0: a_{1} + \cdots + a_{k} = 0
        
        The input params are defined as follows:
            :param df: residuals as the difference between true and prediction
        '''
        names = ['Lagrange multiplier statistic', 'p-value',
            'f-value', 'f p-value']

        # Run Breusch Pagan test:
        test = het_breuschpagan(df.residual, 
                                df[features])

        return print(lzip(names, test))
        
    def homoscedasticity_test_white(self,
                                   df,
                                   features):
        '''
        Test for homoscedasticity in the residuals.
        H0: Homoscedasticity is present
        If pvalue is small, we reject H0 and
        conclude that we have a hetroscedasticity
        problem in our data.
        .. math::
            u^2 = a_{0} + a_{1}*x_{1} + a_{2}*x_{2} + 
                a_{3}*(x_{1}*x_{2}) \cdots + v
            H0: a_{1} + \cdots + a_{k} = 0
        '''

        # add constant
        df = add_constant(df).fillna(0)

        # run white test
        white_test_results = het_white(df.residual, df[['const']+features])
        labels = ['LM-Stat', 'LM p-val', 'F-Stat', 'F p-val'] 
        #print results
        return print(dict(zip(labels, white_test_results)))

    
    def residual_plot(self,
                      fittedvalues,
                      residuals, 
                      ax=None):
        '''
        Plots the fitted values vs residuals.
        (Roughly) Horizontal red line is an indicator
        that the residual has a linear pattern.
        '''
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
                    x=fittedvalues,
                    y=residuals,
                    lowess=True,
                    scatter_kws={'alpha': 0.5},
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                    ax=ax
        )
        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax
    
    def scale_location_plot(self,
                            df,
                            features,
                            residuals,
                            fitted_values,
                            ax=None):
        '''
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        '''
        if type(residuals) is not np.ndarray: 
            residuals = np.array(residuals)
            
        if type(fitted_values) is not np.ndarray: 
            fitted_values = np.array(fitted_values)
            
        leverage, res_studentized_internal = self.__residuals_studentized_internal(df,
                                        residuals,
                                        features)
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(res_studentized_internal))

        ax.scatter(fitted_values, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=fitted_values,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)
        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(fitted_values[i], residual_norm_abs_sqrt[i]),
                color='C3')
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax
    
    def qq_plot(self,
            df,
            features,
            residuals,
            ax=None):
        '''
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        '''
        leverage, res_studentized_internal = self.__residuals_studentized_internal(df,
                                            residuals,
                                            features)
        
        if ax is None:
            fig, ax = plt.subplots()
        QQ = ProbPlot(res_studentized_internal)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(res_studentized_internal)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], res_studentized_internal[i]),
                ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax
    
    def leverage_plot(self,
                  df,
                  features,
                  residuals,
                  fittedvalues,
                  y_true,
                  ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        leverage, res_studentized_internal = self.__residuals_studentized_internal(df,
                                        residuals,
                                        features)
        
        
        cooks_distance = self.__cooks_distance(leverage,
                                               features,
                                               residuals,
                                               fittedvalues,
                                               y_true)
        
        
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            leverage,
            res_studentized_internal,
            alpha=0.5);

        sns.regplot(
            x=leverage,
            y=res_studentized_internal,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(leverage[i], res_studentized_internal[i]),
                color = 'C3')

        xtemp, ytemp = self.__cooks_dist_line(0.5, features, leverage) # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self.__cooks_dist_line(1, features, leverage) # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.set_xlim(0, max(leverage)+0.01)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return ax
    
    def show_all_classic_diagnostic_plots(self, 
                       df,
                       residuals,
                       fittedvalues,
                       features,
                       y_true,  
                       plot_context='seaborn-paper'):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
            self.residual_plot(fittedvalues,residuals,ax=ax[0,0])
            self.qq_plot(df,features,residuals,ax=ax[0,1])
            self.scale_location_plot(df,features,residuals,fittedvalues,ax=ax[1,0])
            self.leverage_plot(df, features, residuals, fittedvalues, y_true, ax=ax[1,1])
            plt.show()

        display(self.vif_test(df[features]))
        return fig, ax
