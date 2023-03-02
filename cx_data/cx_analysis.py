import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score


class CXAnalysis:
    def __init__(self):
        pass

    def chained_pre_processing(self, df):
        """
        Applies multiple pre-processing steps
        """

        def drop_first_two_rows(df):
            df = df.iloc[2:].copy()
            return df

        def create_pd_date_format(df):
            return df.assign(
                yyyymm=df["StartDate"].apply(lambda x: pd.to_datetime(x[0:7] + "-01"))
            )

        df = create_pd_date_format(drop_first_two_rows(df))
        return df

    def cross_validation_split(self, df, features, target, split):
        betas = []
        pvalues = []
        test_prediction = []
        test_truth = []
        train_fitted = []
        train_true = []

        for i in range(0, split):
            indx = df.index[
                (int(df.shape[0] / split) * (i)) : (int(df.shape[0] / split) * (i + 1))
            ].copy()
            df_test = df[df.index.isin(indx)].copy()
            df_train = df[~df.index.isin(indx)].copy()

            y_train = df_train[target]
            X_train = df_train[features]
            X_train = sm.add_constant(X_train)

            y_test = df_test[target]
            X_test = df_test[features]
            X_test = sm.add_constant(X_test)

            model = sm.OLS(y_train, X_train)
            results = model.fit()

            betas.append(results.params[1:].values)
            pvalues.append(results.pvalues[1:].values)
            test_prediction.append(results.predict(X_test))
            test_truth.append(y_test)
            train_fitted.append(results.fittedvalues.values)
            train_true.append(y_train.values)

        r2_train = r2_score(np.concatenate(train_true), np.concatenate(train_fitted))
        r2_test = r2_score(np.concatenate(test_truth), np.concatenate(test_prediction))

        X_full = df[features]
        y_full = df[target]
        X_full = sm.add_constant(X_full)

        model = sm.OLS(y_full, X_full)
        results = model.fit()

        betas.append(results.params[1:].values)
        pvalues.append(results.pvalues[1:].values)

        data = {
            "features": features,
            "beta": betas[-1],
            "lower_bound_beta": results.conf_int()[0][1:].values,
            "upper_bound_beta": results.conf_int()[1][1:].values,
            "pvalue": pvalues[-1],
        }
        dfz = pd.DataFrame(data)
        dfz.sort_values("beta", ascending=False, inplace=True)
        residuals = y_full - results.fittedvalues

        return (
            betas,
            pvalues,
            test_prediction,
            test_truth,
            r2_train,
            r2_test,
            residuals,
            results.fittedvalues,
            dfz,
        )
