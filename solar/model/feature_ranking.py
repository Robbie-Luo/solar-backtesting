import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from dcor import distance_correlation

def correlation(x, y, method='pearson'):
    """
    
    Calculates correlation between two Series in a way that is robust to missing values.
    
    Parameters
    ----------
    x : Series or list or ndarray
        One series.
        
    
    y : Series or list or ndarray
        Another series
        
        
    method : str (pearson, spearman, kendall, distance, roc_auc)
        Correlation method to use
        If method is 'roc_auc', then x should be the target, and y should be the feature
        
    Returns
    -------
    
    corr : double
        Correlation calculated between the valid parts of the two input series.
    
    Function stolen from source code of newer Pandas version...
    DataIKU is behind on Pandas and only supports Pearson R correlation
    """

    def get_corr_func(method):
        if method in ['kendall', 'spearman']:
            from scipy.stats import kendalltau, spearmanr
        elif method == 'distance':
            from dcor import distance_correlation
        elif method == 'roc_auc':
            from sklearn.metrics import roc_auc_score
        elif callable(method):
            return method

        def _pearson(a, b):
            return np.corrcoef(a, b)[0, 1]

        def _kendall(a, b):
            rs = kendalltau(a, b)
            if isinstance(rs, tuple):
                return rs[0]
            return rs

        def _spearman(a, b):
            return spearmanr(a, b)[0]

        def _distance(a, b):
            return distance_correlation(a, b)

        def _roc_auc(a, b):
            return max(roc_auc_score(a, b),1-roc_auc_score(a, b))
        
        _cor_methods = {
            'pearson': _pearson,
            'kendall': _kendall,
            'spearman': _spearman,
            'distance': _distance,
            'roc_auc': _roc_auc
        }
        return _cor_methods[method]

    a = x.values
    b = y.values

    valid = pd.notnull(a) & pd.notnull(b)
    if not valid.all():
        a = a[valid]
        b = b[valid]

    if len(a) < 1:
        return np.nan

    f = get_corr_func(method)
    return f(a, b)

class FeatureRanker:
    """
    What does this do? Define class feature. Define object, input dataset , targets and features to be used.
    Why was it written like this?
    How should I use this?
    """
    
    def __init__(self,input_df,targets,features):
        self.dataframe=input_df
        self.targets=targets
        self.features=features
        self.feature_importance=None
        self.feature_importance_less=None
        
    def calc_importance(self):
        results=[]
        
        for target in self.targets:
            for feature in self.features:
                # CJ: not sure if we should dropna here... This may lead to 
                df=self.dataframe.loc[:, ["date",feature,target]].dropna(how="any")

                results.append([feature,target,list(df['date'])[0],list(df['date'])[-1],"filter","Pearson R", correlation(df[feature],df[target], 'pearson')])
                results.append([feature,target,list(df['date'])[0],list(df['date'])[-1],"filter","Spearman R", correlation(df[feature],df[target], 'spearman')])
                results.append([feature,target,list(df['date'])[0],list(df['date'])[-1],"filter","Kendall Tau", correlation(df[feature],df[target], 'kendall')])
                results.append([feature,target,list(df['date'])[0],list(df['date'])[-1],"filter","Distance Correlation", correlation(df[feature],df[target], 'distance')])
        self.feature_importance=pd.DataFrame(results,columns=["variable","target","date_start","date_end","method","metric","score"])
        df = self.feature_importance.groupby(["variable","target"]).apply(lambda df:np.abs(df["score"]).mean()).reset_index()
        self.feature_importance_less = df.rename(columns = {0: "score"})
        
        
    def __get_rank(self,df):
        df=df.sort_values(by="score",ascending=False).reset_index().drop(["target","index"],axis=1)
        df["ranking"]=df.index+1
        return df.set_index("variable")
    
    def __get_list(self,df):
        df=df.sort_values(by="score",ascending=False)
        return df[["variable"]]
        
    def get_importance(self,option="list"):
        """
        What does this do?
        """

        if self.feature_importance is None:
            raise Exception("Feature importance is not calculated yet.")
        if option=="ranking":
            return self.feature_importance_less.groupby("target").apply(self.__get_rank)
        elif option=="list":
            return self.feature_importance_less.groupby("target").apply(self.__get_list)
        elif option=="raw":
            return self.feature_importance
        else:
            raise Exception("Option must be one of 'raw', 'ranking' or 'list'.")