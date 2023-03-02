import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet

import ols_assumptions_check 


class ModelComparison(object):
    
    def __init__(self):
        pass
    
    def normalization_features(self,
                           df,
                           features):
        scaler = MinMaxScaler()
        df[features] = df[features].dropna()

        for feat in features:
            df[feat] = scaler.fit_transform(df[feat].values.reshape(-1, 1))
        return df

    def ols_model(self,
                  df,
              target,
              features,
              time_features,
              dummy_features,
              date,
              level, 
              extra_trend,
                 constant=1):
        if constant==1:
            df['const']=1
            df = df[[date]+[level]+[target]+features+time_features+dummy_features+['const']+[extra_trend]].dropna()

            mod = sm.OLS(df[target], df[features+time_features+['const']+dummy_features])  
            res = mod.fit(cov_type='HAC',cov_kwds={'maxlags':8})

            pvalues = res.pvalues
            coefs = res.params
            preds = res.predict(df[features+time_features+['const']+dummy_features])
        else:
            df = df[[date]+[level]+[target]+features+time_features+dummy_features+[extra_trend]].dropna()

            mod = sm.OLS(df[target], df[features+time_features+dummy_features])  
            res = mod.fit(cov_type='HAC',cov_kwds={'maxlags':8})

            pvalues = res.pvalues
            coefs = res.params
            preds = res.predict(df[features+time_features+dummy_features])
            
        df['pred'] = preds
        df['residual'] = df.pred - df[target]
        df_temp = pd.DataFrame(res.params)
        cols_used = list(df_temp.index)
        print(res.summary())
        return df, [coefs, pvalues], cols_used, res
    
    def me_model(
                self,
                df,
              target,
              features,
              time_features,
              groups,
              date,
              level,
              extra_trend,
              dummy_features=0):
        if dummy_features !=0:
            #df['const']=1
            df = df[[date]+[level]+[target]+features+time_features+dummy_features+[extra_trend]].dropna()

            mod = sm.MixedLM(df[target], 
                       df[features+time_features+dummy_features], 
                       groups=df[groups], 
                       exog_re=df[features]
                     )
            res = mod.fit(reml=False)
            cols_used = features+time_features+dummy_features
        else:
            #df['const']=1
            df = df[[date]+[level]+[target]+features+time_features+[extra_trend]].dropna()

            mod = sm.MixedLM(df[target], 
                       df[features+time_features], 
                       groups=df[groups], 
                       exog_re=df[features]
                     )
            res = mod.fit(reml=False)
            cols_used = features+time_features

        #pvalues = res.pvalues[0:len(features)]
        pvalues = res.pvalues
        coefs = res.params
        #coefs = res.params[0:len(features)]
        preds = res.predict(df[cols_used])
        df['pred'] = preds
        df['residual'] = df.pred - df[target]
        return df, [coefs, pvalues], cols_used, res
    
    def ols_positive(self,
                 df,
                 target,
                 features,
                 time_features,
                 dummy_features,
                 date,
                 level,
                 extra_trend,
                 constant=1):
        if constant==1:
            df = df[[date]+[level]+[target]+features+time_features+dummy_features+[extra_trend]].dropna()
            df['const']=1
            cols_used = features+time_features+['const']+dummy_features
            res = LinearRegression(fit_intercept=False,
                                   positive=True).fit(df[cols_used], 
                                                           df[target])
        else:
            df = df[[date]+[level]+[target]+features+time_features+dummy_features+[extra_trend]].dropna()
            print(df.shape)
            cols_used = features+time_features+dummy_features
            res = LinearRegression(fit_intercept=False,
                                   positive=True).fit(df[cols_used], 
                                                           df[target])
            
        coefs = res.coef_
        preds = res.predict(df[cols_used])
        df['pred'] = preds
        df['residual'] = df.pred - df[target]
        return df, coefs, cols_used, res
    
    def elastic_net(self,
                 df,
                 target,
                 features,
                 time_features,
                 dummy_features,
                 date,
                 level,
                 extra_trend,
                 alpha=1):
        df = df[[date]+[level]+[target]+features+time_features+dummy_features+[extra_trend]].dropna()
        cols_used = features+time_features+dummy_features
        elastic = ElasticNet(alpha=alpha, fit_intercept=False, random_state=0)
        res = elastic.fit(df[cols_used], df[target])
        coefs = res.coef_
        preds = res.predict(df[cols_used])
        df['pred'] = preds
        df['residual'] = df.pred - df[target]
        return df, coefs, cols_used, res
    
    def gaussian_process(self,
                    df,
                     target,
                 features,
                 time_features,
                 dummy_features,
                 date,
                 level,
                 extra_trend):
        df = df[[date]+[level]+[target]+features+time_features+dummy_features+[extra_trend]].dropna()
        cols_used = features+time_features+dummy_features

        res = GaussianProcessRegressor(random_state=0).fit(df[cols_used],
                                                           df[target])
        preds = res.predict(df[cols_used])
        df['pred'] = preds
        df['residual'] = df.pred - df[target]
        coefs = 0
        return df, coefs, cols_used, res

    def counterfactual(self,
                    df,
                    res,
                    features_all,
                    zero_features):
        
        for feat in zero_features:
            df_temp = df.copy()
            df_temp[feat] = df_temp[feat].apply(lambda x: x*0)
            preds_zero = res.predict(df_temp[features_all])
            df['preds_zero_'+feat] = preds_zero
            df['impact_area_'+feat] = df.pred - df['preds_zero_'+feat]

        cols_zero = df.columns[-len(zero_features)*2:]
        
        df_temp = df.copy()
        df_temp[zero_features] = df_temp[zero_features].apply(lambda x: x*0)
        pred_baseline = res.predict(df_temp[features_all])
        df['baseline_sales'] = pred_baseline
        return df, cols_zero
    
    def aggregated_impact(self,
                          df,
                          cols_zero,
                          target,
                          date,
                          extra_trend,
                          trend_behaviour,
                          number_hcps):
        df['trend_season'] = extra_trend
        df['baseline_sales_trend_season'] = df.baseline_sales * df.trend_season * number_hcps
        df['pred_trend_season'] = df.pred * df.trend_season * number_hcps
        
        search_word = 'preds'
        cols_for_impact = [s for s in cols_zero if search_word in s]
        cols_area_impact = list(set(cols_zero) ^ set(cols_for_impact))
        
        for feat in cols_for_impact:
            if trend_behaviour=='stl':
                df['impact_'+feat] = (df.pred_trend_season) / (df[feat]*df.trend_season*number_hcps) - 1
                df['delta_'+feat] = (df.pred_trend_season) - (df[feat]*df.trend_season*number_hcps)
            if trend_behaviour=='linear':
                df['impact_'+feat] = (df.pred+df.trend) / (df[feat]+df.trend_season) - 1
            else: 
                df['impact_'+feat] =  (df.pred) / (df[feat]) - 1

        search_word = 'delta_'
        cols_for_delta = [s for s in df.columns if search_word in s]

        dfx = df.groupby(date)[cols_zero.tolist()+['pred']+[target]+['trend_season']+['baseline_sales']+['baseline_sales_trend_season']+
                              ['pred_trend_season']+cols_for_delta].sum().reset_index()
        
        search_word = 'delta_'
        cols_for_delta = [s for s in dfx.columns if search_word in s]

        for col in cols_for_delta:
            dfx['impactable_sales_'+col] = dfx[col]/dfx.pred_trend_season
            dfx['baseline_share'] = dfx.baseline_sales_trend_season/dfx.pred_trend_season
        
        return dfx, cols_for_impact, cols_for_delta
    
    def summary_plot(self,
                 df,
                 target,
                 country_used,
                 brand_used
            ):
        search_word = 'impactable'
        cols_for_plot_impact = [s for s in df.columns if search_word in s]
        df[cols_for_plot_impact].plot(figsize=(10,4))
        plt.title(country_used+' '+brand_used+': impactable sales')
        plt.show() 

        df.baseline_share.plot(figsize=(10,4))
        plt.title(country_used+' '+brand_used+': Baseline share')
        plt.show()

        df[['pred',target]].plot(figsize=(10,4))
        plt.title(country_used+' '+brand_used+': True vs prediction')
        plt.show()
        
    def print_impact(self,
                     df,
                     date,
                     start_date,
                     cols_for_delta):
        print('Overall impact of all channels: {}'.format(np.sum(df[df[date]>=start_date][cols_for_delta],axis=1).sum()
                                              / np.sum(df[df[date]>=start_date].pred_trend_season)))


        for i in cols_for_delta:
            print('overall impact of channel {} is: {}'.format(i,
                                                                   np.sum(df[df[date]>=start_date][i])/
                                                                   np.sum(df[df[date]>=start_date].pred_trend_season)))
            
    def calculate_impact(
                     self,
                     df,
                     date,
                     start_date,
                     cols_for_delta):
        overall_impact = np.sum(df[df[date]>=start_date][cols_for_delta],axis=1).sum()/ np.sum(df[df[date]>=start_date].pred_trend_season)

        impact_per_channel = []
        for i in cols_for_delta:
             impact_per_channel.append(np.sum(df[df[date]>=start_date][i])/np.sum(df[df[date]>=start_date].pred_trend_season))

        return overall_impact, impact_per_channel
    
    def concatenate_results(self,
                        coef,
                        pvalues,
                        overall_impact,
                        impact_per_channel,
                        errors,
                        i):
        results = np.concatenate([coef[i],
                                  pvalues[i],
                                  impact_per_channel[i],
                                  np.array([overall_impact[i]]),
                                  np.array([errors[i]]),
                   ])
        return results
    
    def concat_all_results(self,
                       features,
                       time_features,
                       beta_coefficients,
                       pvalues,
                       impact_per_channel,
                       impact_overall,
                       mape_loss,
                       dates_to_loop
                       ):
        beta_cols = [feat+'_beta' for feat in features+time_features]
        pvalue_cols = [feat+'_pvalue' for feat in features+time_features]
        impact_cols = [feat+'_impact' for feat in features]

        data = {'Features': beta_cols+pvalue_cols+impact_cols+['impact_overall']+['mape'],
            dates_to_loop[0][1]: self.concatenate_results(beta_coefficients,
                            pvalues,
                            impact_overall,
                            impact_per_channel,
                            mape_loss,
                            0),

            dates_to_loop[1][1]: self.concatenate_results(beta_coefficients,
                            pvalues,
                            impact_overall,
                            impact_per_channel,
                            mape_loss,
                            1),
            dates_to_loop[2][1]: self.concatenate_results(beta_coefficients,
                            pvalues,
                            impact_overall,
                            impact_per_channel,
                            mape_loss,
                            2),
            dates_to_loop[3][1]: self.concatenate_results(beta_coefficients,
                            pvalues,
                            impact_overall,
                            impact_per_channel,
                            mape_loss,
                            3),
           }
        dfz = pd.DataFrame(data)
        return dfz
    
    def mape(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mape = np.mean(abs(y_true-y_pred))
        return mape

    def smape(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        smape = np.mean(abs((y_true-y_pred))/(abs(y_true)+abs(y_pred)))   
        return smape
    
    def filter_dates(self, df, min_date, end_date):
        df = df[(df.yyyymm>=min_date)&
             (df.yyyymm<=end_date)].copy()
        return df
    
    def run_mode_training(self,
                          df,
                          target,
                          time_features,
                          features,
                          country,
                          brand,
                          area,
                          hcp_scaler,
                          country_column,
                          brand_column,
                          date_column,
                          dates_to_loop,
                          selected_algo,
                          trend_variable,
                          show_classic_model_diagnostics=0,
                          show_additional_model_diagnostics=0,
                          show_summary_plots=0,
                          show_vif=0,
                          constant=0,
                          area_dummy=0):
        
        cl = ols_assumptions_check.ols_check(area, date_column)
    
        mdb = df
        country = country
        brand = brand
        selected_algo = selected_algo
        country_column = country_column
        brand_column = brand_column

        trend_variable = trend_variable 
        date = date_column
        trend_ts_method = 'stl'
        constant = constant

        beta_coefficients = []
        pvalues = []
        mape_loss = []
        impact_overall = []
        impact_per_channel = []
        cph_share = []

        for d in dates_to_loop:
            df = self.filter_dates(mdb.query("{} == '{}' and {} == '{}'".format(country_column, country, brand_column, brand)),d[0],d[1])
            temp = df[features+time_features+[hcp_scaler]+[area]+[target]+[hcp_scaler]].dropna()
            #hcp_temp = temp[hcp_scaler].values
            hcp_temp=1
            
            area_cat = pd.get_dummies(df[area].unique(), drop_first=True)
            area_cat = area_cat.columns.values.tolist() 

            if selected_algo =='ols_positive':
                if area_dummy==0:
                    model = self.ols_positive(df,
                              target,
                              features,
                              time_features,
                              [],
                              date,
                              area,
                              trend_variable,
                              constant=constant)
                else:
                    model = self.ols_positive(df,
                              target,
                              features,
                              time_features,
                              area_cat,
                              date,
                              area,
                              trend_variable,
                              constant=constant)
                        
            else:
                if area_dummy==0:
                    model = self.ols_model(df,
                                  target,
                                  features,
                                  time_features,
                                  [],
                                  date,
                                  area,
                                  trend_variable,
                                  constant=constant)
                else:
                    model = self.ols_model(df,
                                  target,
                                  features,
                                  time_features,
                                  area_cat,
                                  date,
                                  area,
                                  trend_variable,
                                  constant=constant)
                    
            dfx, inference, cols_used, res = model
            dfx2,cols_zero = self.counterfactual(dfx,
                               res,
                               cols_used,
                               features)

            dfx3, cols_for_agg, cols_for_delta = self.aggregated_impact(dfx2,
                                  cols_zero,
                                  target,
                                  date,
                                  dfx[trend_variable].values,
                                  trend_ts_method,
                                  hcp_temp)
            print(d[0],d[1])

            impact_overall_temp, impact_per_channel_temp = self.calculate_impact(dfx3,
                                                                     date_column,
                                                                     '2021-06-01',
                                                                     cols_for_delta)
            
            if selected_algo=='ols_positive':
                beta_coefficients.append(inference[:len(features+time_features)])
                pvalues.append(np.array(len(inference[:len(features+time_features)])*[None]))
            else:
                beta_coefficients.append(inference[0][:len(features)+len(time_features)].values)
                pvalues.append(inference[1][:len(features)+len(time_features)].values)
            mape_loss.append((self.mape(dfx2[target], dfx2.pred)))
            impact_overall.append(impact_overall_temp)
            impact_per_channel.append(impact_per_channel_temp)
            
            if show_summary_plots!=0:
                if d[1]==dates_to_loop[-1][1]:
                     self.summary_plot(dfx3,target,country,brand)

            if show_classic_model_diagnostics!=0:
                if d[1]==dates_to_loop[-1][1]:
                    cl.show_all_classic_diagnostic_plots(dfx, dfx.residual, dfx.pred, cols_used)

            if show_additional_model_diagnostics!=0:
                if d[1]==dates_to_loop[-1][1]:
                    cl.show_additional_residual_plots(dfx, dfx.residual, area, 10)
                    print('Durbin Watson for pooled sample:{}'.format(cl.serial_correlation_check(dfx,full=0)))
                    
            if show_vif!=0:
                if d[1]==dates_to_loop[-1][1]:
                    df_vif = cl.vif_test(dfx[cols_used])
                    print(df_vif)

        dfz = self.concat_all_results(
                         features=features,
                         time_features=time_features,
                         beta_coefficients=beta_coefficients,
                         pvalues=pvalues,
                         impact_per_channel=impact_per_channel,
                         impact_overall=impact_overall,
                         mape_loss=mape_loss,
                         dates_to_loop=dates_to_loop)

        return dfz