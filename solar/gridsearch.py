import numpy as np, pandas as pd
from solar.feature_selection_methods import *
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm_notebook as tqdm
pd.set_option('mode.chained_assignment', None)

def parse_nested_dict(d):
    dl = {}
    for k in d:
        obj, arg = k.split('.')
        if not obj in dl:
            dl[obj] = {}
        dl[obj][arg] = d[k]
    return dl

def parse_hyperparam_grid(d):
    import itertools
    return list((dict(zip(d, x)) for x in itertools.product(*d.values())))

class GridSearch:
    
    def __init__(self, sig_model_objs, param_dict, sig_backtester):
        self.sig_model_objs = sig_model_objs
        self.sig_backtester = sig_backtester
        self.param_list = parse_hyperparam_grid(param_dict)
        
    def _create_target(self, df, target_var, time_horizon, task, est=None):
        
        if task == 'classification':
            raise Exception("Not implemented yet.")
            
        elif task == 'ordinal_regression':
            
            if not est:
                raise Exception("Discretizer not found.")
            
            df['target'] = -(df[target_var].diff(-time_horizon))
            df = df.iloc[:-time_horizon]
            df.loc[:,'target'] = est.fit_transform(df.loc[:, "target"].dropna().values.reshape(-1,1))
            
        elif task == 'regression':
            raise Exception("Not implemented yet.")
            
        elif task == 'abs_ordinal_regression':
            if not est:
                raise Exception("Discretizer not found.")
            
            df['target'] = abs(-(df[target_var].diff(-time_horizon)))
            df = df.iloc[:-time_horizon]
            df.loc[:,'target'] = est.fit_transform(df.loc[:, "target"].dropna().values.reshape(-1,1))
            
        return df
    
    def run(self, input_df, time_horizons, feature_lists, max_iterations=36, train_length=60, test_length=1, normalize=True, target_var="ZuivelNL - Dutch butter quotation", task='classification', verbose=0):
        predictions = []
        metrics = []
        for param in tqdm(self.param_list):
            for time_horizon in time_horizons:
                parsed_param = parse_nested_dict(param)
                
                # Here starts the feature selection.
                # This implementation is not flexible, which might need to be improved later.
                '''
                feature_df = input_df.copy()

                feature_df['Target'] = -(feature_df['ZuivelNL - Dutch butter quotation'].diff(-time_horizon))
                feature_df = feature_df.iloc[:-time_horizon]
                est = KBinsDiscretizer(n_bins= parsed_param['KBinsDiscretizer']['n_bins'] , encode='ordinal', strategy='uniform')
                feature_df['target'] = est.fit_transform((feature_df[['Target']]))
                input_feat = feature_df.columns.tolist()
                input_feat.remove('target')
                input_feat.remove('Target')
                input_feat.remove('date')
                feature_df[input_feat] = feature_df[input_feat].ffill().diff(time_horizon)
                v_feats = remove_invalid_features(feature_df)
                v_feats = eliminate_multicolinear_features(feature_df,v_feats)
                feature_lists = boruta_feature_selection(feature_df, v_feats, RandomForestRegressor(), max_iter=500)
                '''
                
                feature_df = input_df.copy()
                est = KBinsDiscretizer(n_bins= parsed_param['KBinsDiscretizer']['n_bins'] , encode='ordinal', strategy='uniform')
                feature_df = self._create_target(feature_df, target_var, time_horizon, task, est)
                input_feat = feature_lists[time_horizon]
                v_feats = list(set(remove_invalid_features(feature_df))&set(input_feat))
                v_feats = eliminate_multicolinear_features(feature_df,v_feats)
                feature_list = boruta_feature_selection(feature_df, v_feats, RandomForestRegressor(), max_iter=500)
                
                # Here ends the feature selection.
                bt_kwargs = {
                    'input_df': input_df,
                    'time_horizon': time_horizon,
                    'feature_list': feature_list,
                    'max_iterations': max_iterations,
                    'target_var': target_var,
                    'task': task,
                }
                for k in self.sig_model_objs:
                    if self.sig_model_objs[k].__name__ in parsed_param.keys():
                        bt_kwargs[k] = self.sig_model_objs[k](**parsed_param[self.sig_model_objs[k].__name__])
                    else:
                        bt_kwargs[k] = self.sig_model_objs[k]()
                bt_clf = self.sig_backtester(**bt_kwargs)
                pred_df, acc_df = bt_clf.run(train_length, test_length, normalize)
                pred_df['time_horizon'] = time_horizon
                pred_df['model'] = task
                metric_df = bt_clf.evaluate()[1]
                if 'ordinal_regression' in task:
                    bin_edges = bt_clf.est.bin_edges_[0]
                    pred_df['bin_edges'] = str(bin_edges)
                    metric_df['bin_edges'] = str(bin_edges)
                pred_df['hyperparam'] = str(param)
                metric_df['hyperparam'] = str(param)
                predictions.append(pred_df)
                metrics.append(metric_df)
                if verbose > 0:
                    print(f"Finish testing hyperparameter: {param}, time horizon: {time_horizon}.")
        return pd.concat(predictions), pd.concat(metrics)