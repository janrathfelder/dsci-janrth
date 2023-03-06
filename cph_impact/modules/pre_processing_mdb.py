import numpy as np
import pandas as pd
import holidays
import calendar
import datetime

from typing import List

class PreProcessing(object):
    '''
    Different functions to apply cleaning and pre-processing aimed to prepare the data frame 
    for impact attribution for each marketing channel.
    '''
    
    def __init__(self):
        pass
    
    def scaling(self,
                df: pd.DataFrame,
                denominator: pd.Series,
                feature: pd.Series) -> pd.DataFrame:
        '''
        Scales a feature using the specified denominator.
        '''
        df[feature] = df[feature]/df[denominator]
        return df
    
    def sorting(self, 
                df: pd.DataFrame, 
                grouping: List[str], 
                date: str) -> pd.DataFrame:
        '''
        Sorts a data frame by a grouping and date.
        '''
        df = df.sort_values(grouping+[date]).reset_index(drop=True).copy()
        return df

    def create_date_format(self, 
                           df: pd.DataFrame,
                           original_date_column: str) -> pd.DataFrame:
        '''
        Creates a pandas datetime object from a date string.
        Very specifiy to the output of the mdb notebook yrmo column.
        '''
        df['yyyymm'] = df[original_date_column].apply(lambda x: str(x)[:4]+'-'+str(x)[4:]+'-01')
        df['yyyymm'] = pd.to_datetime(df.yyyymm)
        return df
    
    def remove_zero(self, 
                    df: pd.DataFrame, 
                    variable: str, 
                    grouping: List[str], 
                    date: str, 
                    min_date: str, 
                    max_date: str, 
                    threshold: float) -> pd.DataFrame:
        '''
        Removes areas that are below threshold for specified variable.
        Filtering is applied <= threshold, meaning is threshold=0, then areas
        with sum of zero for specified variable are removed.
        '''
        gr = df[(df[date]>=min_date)&(df[date]<=max_date)].groupby(grouping)[variable].sum().reset_index()
        gr['flag'] = np.where(gr[variable]<=threshold,1,0)
        gr = gr[gr.flag==0]
        df = df.merge(gr[grouping+['flag']], on=grouping, how='inner')
        df.drop('flag', axis=1, inplace=True)
        return df
        
    def remove_min_value(self, 
                         df: pd.DataFrame, 
                         variable: str, 
                         grouping: List[str], 
                         date: str, 
                         min_date: str, 
                         max_date: str, 
                         threshold: float) -> pd.DataFrame:
        '''
        Finds the minimum value per group for a variable and removes those groups, that
        are below a threshold value.
        '''
        gr = df[(df[date]>=min_date)&(df[date]<=max_date)].groupby(grouping)[variable].min().reset_index()
        gr['flag'] = np.where(gr[variable]<threshold,1,0)
        gr = gr[gr.flag==0]
        df = df.merge(gr[grouping+['flag']], on=grouping, how='inner')
        df.drop('flag', axis=1, inplace=True)
        return df
    
    def remove_max_value(self, 
                         df: pd.DataFrame, 
                         variable: str, 
                         grouping: List[str], 
                         date: str, 
                         min_date: str, 
                         max_date: str, 
                         threshold: float) -> pd.DataFrame:
        '''
        Finds the maximum value per group for a variable and removes those groups, that
        are above a threshold value.
        '''
        gr = df[(df[date]>=min_date)&(df[date]<=max_date)].groupby(grouping)[variable].max().reset_index()
        gr['flag'] = np.where(gr[variable]>threshold,1,0)
        gr = gr[gr.flag==0]
        df = df.merge(gr[grouping+['flag']], on=grouping, how='inner')
        df.drop('flag', axis=1, inplace=True)
        return df
    
    def remove_na(self, 
                  df: pd.DataFrame, 
                  variable: str, 
                  grouping: List[str], 
                  date: str, 
                  min_date: str, 
                  max_date: str) -> pd.DataFrame:
        '''
        Removes groups that have inf values in the specified variable.
        '''
        gr = df[(df[date]>=min_date)&(df[date]<=max_date)].groupby(grouping)[variable].sum().reset_index()
        gr['flag'] = np.where(gr[variable]==np.inf,1,0)
        gr = gr[gr.flag==0]
        df = df.merge(gr[grouping+['flag']], on=grouping, how='inner')
        df.drop('flag', axis=1, inplace=True)
        return df
    
    def time_index(self, 
                   date: str, 
                   df: pd.DataFrame) -> pd.DataFrame:
        '''
        Creates month, year and year+month from pandas datetime object.
        '''
        df['month'] = df[date].dt.month
        df['year'] = df[date].dt.year
        df['year_month'] = df.year+df.month
        return df
    
    def sales_potential(self, 
                        df: pd.DataFrame, 
                        date: str, 
                        grouping: List[str], 
                        sales: str, 
                        max_date: str) -> pd.DataFrame:
        '''
        Creates 5 groups based on sales percentiles for each country-brand combination.
        '''
        gr_sales = df[(df[date]<=max_date)].groupby([grouping])[sales].sum().reset_index()
        sales_pet = []

        for c in gr_sales.country_cd.unique():
            for b in gr_sales.brand_nm.unique():
                gr_temp = gr_sales[(gr_sales.brand_nm==b)&(gr_sales.country_cd==c)].copy()
                gr_temp['sales_potential'] = np.where(gr_temp.sales_unit <= np.percentile(gr_temp.sales_unit,20),'xs',
                                  np.where(gr_temp.sales_unit <= np.percentile(gr_temp.sales_unit,40),'s',
                                  np.where(gr_temp.sales_unit <= np.percentile(gr_temp.sales_unit,60),'m', 
                                  np.where(gr_temp.sales_unit <= np.percentile(gr_temp.sales_unit,80),'l', 
                                  'xl'))))
                sales_pet.append(gr_temp['sales_potential'].values)
        
        gr_sales['sales_potential'] = np.concatenate(sales_pet)   
        df = df.merge(gr_sales[grouping+['sales_potential']], 
                                on=grouping)
        return df

    def shifting(self,
                 df: pd.DataFrame,
                 grouping: str,
                 date: str,
                 columns: List[str],
                 shifts: List[int]) -> pd.DataFrame:
        '''
        Shifts columns thereby creating new columns.
        '''
        df = self.sorting(df, grouping, date)
        for cols in columns:
            for shift in shifts:
                df[cols+'_shift'+str(shift)] = df.groupby(grouping)[cols].shift(shift)
        return df
    
    def rolling_window(self,
                       df: pd.DataFrame,
                       grouping: str,
                       date: str,
                       features: List[str],
                       windows: List[int]) -> pd.DataFrame:
        '''
        Creates rolling windows of different lengths depending on the specified
        window lenghts for variables.
        '''
        df = self.sorting(df, grouping, date)
        for feat in features:
            for w in windows:
                df[feat+'_'+str(w)+'mean'] = df.groupby(grouping)[feat].rolling(w).mean().values
        return df
    
    def filter_dates(self,
                     df: pd.DataFrame,
                     date: str,
                     min_date: str,
                     end_date: str) -> pd.DataFrame:
        '''
        Applies filtering on a data frame.
        '''
        df = df[(df[date]>=min_date)&
             (df[date]<=end_date)].copy()
        return df
    
    def replace_missing_hcp_count_bfill(self,
                                        df: pd.DataFrame,
                                        hcp_variable: str,
                                        grouping: List[str],
                                        date: str) -> pd.DataFrame:
        '''
        Applies backward filling per group for specified variable.
        '''
        df['id_key'] = df.groupby(grouping).ngroup()
        df.sort_values(['id_key',date], inplace=True)
        hcp_corrected = []
        for k in df.id_key.unique():
                hcp_corrected.append(df[(df.id_key==k)][hcp_variable].replace(to_replace=0, method='bfill').values)
        df[hcp_variable] = np.concatenate(hcp_corrected)
        df.drop('id_key', axis=1, inplace=True)
        return df
    
    def replace_missing_hcp_count_ffill(self,
                                        df: pd.DataFrame,
                                        hcp_variable: str,
                                        grouping: List[str],
                                        date: str) -> pd.DataFrame:
        '''
        Applies forward filling per group for specified variable.
        '''
        df['id_key'] = df.groupby(grouping).ngroup()
        df.sort_values(['id_key',date], inplace=True)
        hcp_corrected_new = []
        for k in df.id_key.unique():
                hcp_corrected_new.append(df[(df.id_key==k)][hcp_variable].replace(to_replace=0, method='ffill').values)
        df[hcp_variable] = np.concatenate(hcp_corrected_new)
        df.drop('id_key', axis=1, inplace=True)
        return df
    
    def remove_low_marketing_activity(self,
                                     df: pd.DataFrame,
                                     date: str,
                                     grouping: List[str],
                                     features: List[str],
                                     start_date: str,
                                     remove_everywhere=1) -> pd.DataFrame:
        '''
        Removes areas with low marketing activity. If remove_everywhere=1, then all areas
        that have 0 for all of the features during the specified period. Otherwise areas that
        have zero activity during the specified period in one out of all features are removed.
        '''
        relevant_features = features

        df['id_key'] = df.groupby(grouping).ngroup()
        mdb_gr = df[df[date]>=start_date].groupby('id_key')[relevant_features].mean().reset_index()

        mdb_gr['prod_features'] = mdb_gr[relevant_features].prod(axis=1)
        mdb_gr['sum_features'] = mdb_gr[relevant_features].sum(axis=1)
        
        key_prod_low = mdb_gr[mdb_gr.prod_features==0].id_key.unique()
        key_sum_low = mdb_gr[mdb_gr.sum_features==0].id_key.unique()
        
        if remove_everywhere==1:
            df = df[~df.id_key.isin(key_sum_low)].copy()
        else:
            df = df[~df.id_key.isin(key_prod_low)].copy()
        df.drop('id_key', axis=1, inplace=True)
        return df
    
    def categorical(self, 
                    df: pd.DataFrame, 
                    grouping: str) -> pd.DataFrame:
        '''
        Creates dummy variables (OHE style) from specified categorical variable.
        '''
        df = pd.concat((
             df,
             pd.get_dummies(df[grouping], drop_first=True)), axis=1)
        return df
    
    def number_of_zero_sales(self,
                             df: pd.DataFrame,
                             variable: str,
                             grouping: List[str],
                             date: str,
                             min_date: str,
                             max_date: str,
                             threshold=.25) -> pd.DataFrame:
        df['id_key'] = df.groupby(grouping).ngroup()

        gr = df[(df[date]>=min_date)&(df[date]<=max_date)].groupby('id_key')[variable].agg(lambda x: x.eq(0).sum()).reset_index()
        gr['n_month'] = df[(df[date]>=min_date)&(df[date]<=max_date)][date].nunique() 
        gr['share'] = gr[variable]/gr.n_month
        gr['flag'] = np.where(gr.share>=threshold,1,0)
        gr = gr[gr.flag==0]
        df = df.merge(gr[['id_key','flag']], on='id_key', how='inner')
        df.drop('id_key', axis=1, inplace=True)
        df.drop('flag', axis=1, inplace=True)
        return df
    
    def find_holidays(self,
                      start_year: int, 
                      end_year: int, 
                      country_list: List[str]) -> pd.DataFrame:
        '''
        Finds the amount of bank holidays per country. For GB the holidays are counted because
        if a holidays falls on a weekend the followwing Monday will be a holiday. For other 
        countries this is not implemented.
        
        Country list needs to be specified according to the country abbrivation understood by the 
        holiday library:
        https://pypi.org/project/holidays/
        '''
        start = start_year
        end = end_year

        month_year = []
        c = []

        countrylistLoop = country_list

        for country in countrylistLoop:
            hd = sorted(holidays.CountryHoliday(country, years=np.arange(start,end+1,1)).items())
            for i,v in enumerate(hd):
                if country!='GB':
                    if hd[i][0].isoweekday() < 6:
                        if len(str(hd[i][0].month)) >= 2:
                            month_year.append(str(hd[i][0].year)+'-'+str(hd[i][0].month)+'-'+'01') 
                            c.append(country)
                        else:
                            month_year.append(str(hd[i][0].year)+'-'+'0'+str(hd[i][0].month)+'-'+'01') 
                            c.append(country)
                else:
                    if len(str(hd[i][0].month)) >= 2:
                        if hd[i][0].isoweekday() == 6:
                            month_year.append(str(hd[i][0].year)+'-'+str((hd[i][0]+datetime.timedelta(days=2)).month)+'-'+'01') 
                            c.append(country)
                        elif hd[i][0].isoweekday() == 7:
                            month_year.append(str(hd[i][0].year)+'-'+str((hd[i][0]+datetime.timedelta(days=1)).month)+'-'+'01') 
                            c.append(country)
                        else:
                            month_year.append(str(hd[i][0].year)+'-'+str(hd[i][0].month)+'-'+'01')
                            c.append(country)

                    else:
                        if hd[i][0].isoweekday() == 6:
                            month_year.append(str(hd[i][0].year)+'-'+'0'+str((hd[i][0]+datetime.timedelta(days=2)).month)+'-'+'01')
                            c.append(country)
                        elif hd[i][0].isoweekday() == 7:
                            month_year.append(str(hd[i][0].year)+'-'+'0'+str((hd[i][0]+datetime.timedelta(days=1)).month)+'-'+'01')
                            c.append(country)
                        else:
                            month_year.append(str(hd[i][0].year)+'-'+str(hd[i][0].month)+'-'+'01') 
                            c.append(country)


        data = {'yyyymm':month_year,
                'country':c}
        df = pd.DataFrame(data)
        df['n']=1
        df2 = df.groupby(['yyyymm','country'])['n'].sum().reset_index().sort_values(['country','yyyymm'])
        df2.rename(columns={'n':'holidays'}, inplace=True)
        return df2

    def find_week_days(self, 
                       start_year: int, 
                       end_year: int) -> pd.DataFrame:
        '''
        Finds the amount of week days pers month per calender year(s).
        '''
        cal = calendar.Calendar()
        years = np.arange(start_year,end_year+1,1)

        month_year_list = []
        working_days = []

        for y in years:
            for m in np.arange(1,13):
                weekday_count = 0
                for week in cal.monthdayscalendar(y, m):
                    for i, day in enumerate(week):
                        # not this month's day or a weekend
                        if day == 0 or i >= 5:
                            continue
                        # or some other control if desired...
                        weekday_count += 1
                if len(str(m)) > 1:
                    month_year_list.append(str(y)+'-'+str(m)+'-01')
                else:
                    month_year_list.append(str(y)+'-0'+str(m)+'-01')
                working_days.append(weekday_count)
        data = {'yyyymm':month_year_list,
            'week_days':working_days}
        df = pd.DataFrame(data)
        return df
    
    def working_days(self,
                     start_year: int, 
                     end_year: int, 
                     country_list: List[str]) -> pd.DataFrame:
        '''
        Finds the amount of working days per month and per country for the specified period by 
        substracting the national wide bank holidays from the working days. For GB the holidays are 
        counted because if a holidays falls on a weekend the followwing Monday will be a holiday. For 
        other countries this is not implemented.  
        '''
        df_holiday = self.find_holidays(start_year, end_year, country_list)
        df_weekd = self.find_week_days(start_year, end_year)

        frames = []
        for c in country_list:
            df_weekd['country'] = c
            temp = df_weekd.copy()
            frames += [temp]
        df_final_weekd = pd.concat(frames)
        df_final_weekd

        df = df_final_weekd.merge(df_holiday, on=['yyyymm', 'country'], how='left')
        df = df.fillna(0)
        df['working_days'] = df.week_days - df.holidays
        df['yyyymm'] = pd.to_datetime(df.yyyymm)
        return df
    
    
    
    