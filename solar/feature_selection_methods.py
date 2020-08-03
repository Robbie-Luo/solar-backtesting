import numpy as np, pandas as pd
from solar.model.feature_ranking import correlation

def remove_invalid_features(df, min_num_valid=100):
    v_check = df.ffill().isna().sum() < len(df) - min_num_valid
    v_feats = v_check[v_check].index.tolist()
    if 'date' in v_feats:
        v_feats.remove('date')
    if 'target' in v_feats:
        v_feats.remove('target')
    return [f for f in v_feats if len(df[f].unique()) > 1]

def feature_clustering(df, threshold_coef, groupwise_distance):
    from scipy.cluster import hierarchy as spc
    corr_matrix = df.corr().abs()
    pdist = spc.distance.pdist(1 - corr_matrix)
    linkage = spc.linkage(pdist, method=groupwise_distance)
    idx = spc.fcluster(linkage, (1 - threshold_coef) * pdist.max(), 'distance')
    groups = pd.DataFrame({'group':idx, 'feat_name':corr_matrix.columns })
    return groups

def eliminate_multicolinear_features(df,v_feats, threshold_coef=0.77, groupwise_distance='single'):
    input_df = df[v_feats].ffill().dropna()
    input_df = input_df[[f for f in v_feats if len(input_df[f].unique()) > 1]]
    groups = feature_clustering(input_df, threshold_coef, groupwise_distance)
    groups['target_corr'] = groups.feat_name.map(lambda x: np.abs(correlation(df.target, df[x])))
    groups = groups.sort_values('target_corr').drop_duplicates('group',keep = 'last')
    return groups.feat_name.tolist()

def boruta_feature_selection(df, v_feats, model, max_iter = 100, ):
    from boruta import BorutaPy
    feat_selector = BorutaPy(model, n_estimators='auto',perc=100,max_iter= max_iter,alpha= 0.05, verbose=0, random_state=1)
    df = df[v_feats+['target']].ffill().dropna()
    feat_selector.fit(df[v_feats].values,df['target'].values)
    return df[v_feats].iloc[:,feat_selector.support_].columns.tolist()

def VIF_mat(df,v_feats,n=30):
    
    a = len(v_feats)-n

    df = df[v_feats].ffill().dropna()
    for i in np.arange(1, a, 1):
        vif = pd.DataFrame()
        vif["variables"] = v_feats
        try:
            vif_res = np.linalg.inv(df[v_feats].corr().fillna(1).values).diagonal()
        except Exception as e:
            if isinstance(e, LinAlgError):
                vif_res = np.linalg.pinv(df[v_feats].corr().fillna(1).values).diagonal()
            else:
                break
        vif["VIF"] = vif_res
        vif = vif.sort_values(by=['VIF'])
        v_feats = vif.iloc[:-1,:]['variables'].tolist()
        df = df[v_feats]
        i = i+1
        
    return v_feats
    
def VIF(df,v_feats,n):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    a = len(v_feats)-n

    df = df[v_feats].ffill().dropna()
    for i in np.arange(1, a, 1):
        vif = pd.DataFrame()
        vif["variables"] = v_feats
        vif["VIF"] = [variance_inflation_factor(df.values, j) for j in range(len(v_feats))]
        vif = vif.sort_values(by=['VIF'])
        v_feats = vif.iloc[:-1,:]['variables'].tolist()
        df = df[v_feats]
        i = i+1
    return v_feats