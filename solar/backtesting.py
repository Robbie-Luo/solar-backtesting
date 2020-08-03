"""
This class can be used for both of the classification and regression tasks.
Example usage can be found in https://10.189.40.171:8443/projects/Solar_Development/notebooks/jupyter/backtesting/
"""

# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
from solar.time_handling import add_months
import solar
from solar.metrics import *
from sklearn.preprocessing import KBinsDiscretizer

class Backtesting():
    """
    A class for fast backtesting
    
    Author: Robbie
    Last Modified: 2020/7/27
    Updates: Using increasing training length by default
    
    Args:
        input_df: input data
        model: backtesting model
        time_horizon: time horizon for prediction
        feature_list: selected features for training
    """
    def __init__(self, input_df, model, time_horizon, feature_list, max_iterations=36, target_var="ZuivelNL - Dutch butter quotation", task = 'classification', est=KBinsDiscretizer(n_bins= 5 , encode='ordinal', strategy='uniform')):
        self.input_df = input_df
        self.model =  model
        self.time_horizon = time_horizon
        self.feature_list = feature_list
        self.target_var = target_var
        self.est = est
        self.task = task
        self.max_iterations = max_iterations
        self.normalize = True
        assert self.task in ['classification','regression','ordinal_regression','abs_ordinal_regression']
        self.preprocess()
        
    def preprocess(self):
        input_df = self.input_df.copy()
        v_check = input_df.isna().sum() < len(input_df) - self.max_iterations - 2*self.time_horizon
        feats_set = set(v_check[v_check].index)
        feature_list = [feat for feat in self.feature_list if feat in feats_set]
        feature_list = list(set(feature_list + [self.target_var]))
        input_df = input_df[["date"]+feature_list].ffill()
        time_horizon = self.time_horizon
        if self.task == 'classification':
            clf_target = -np.sign(input_df[self.target_var].diff(-time_horizon)).fillna(0).astype('int')
            clf_target.loc[clf_target==0] = np.nan
            input_df.loc[:, "TARGET"] = clf_target.ffill()
            
        if self.task == 'regression' or self.task == 'ordinal_regression':
            input_df.loc[:, "TARGET"] = -input_df[self.target_var].diff(-time_horizon)
            self.est.fit(input_df.loc[:, "TARGET"].dropna().values.reshape(-1,1))
        if self.task == 'abs_ordinal_regression':
            input_df.loc[:, "TARGET"] = abs(-input_df[self.target_var].diff(-time_horizon))
            self.est.fit(input_df.loc[:, "TARGET"].dropna().values.reshape(-1,1))
           
        input_df.loc[:, "feature_date"] = input_df["date"]
        input_df.loc[:, "target_date"] = input_df.apply(lambda df: add_months(df["date"],time_horizon),axis=1)

        input_df.drop("date", axis=1, inplace=True)
        self.feature_df = input_df.drop(['TARGET', 'target_date', 'feature_date'],axis=1).ffill()
        self.target_df = input_df['TARGET'].ffill()
        self.date_df = input_df[['target_date', 'feature_date']].ffill()
        
    def split_dataset(self, df, iteration, sliding):
        test_offset = self.time_horizon + iteration 
        train_offset = test_offset + self.test_length + self.time_horizon -1
        if sliding:
            train_df = df.iloc[-train_offset-self.train_length:-train_offset]
        else:
            train_df = df.iloc[:-train_offset]
        test_df =  df.iloc[-test_offset-self.test_length:-test_offset]
        return train_df,test_df
    
    def train(self, train_X, test_X, train_Y, test_Y, train_T, test_T):
        if self.normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            train_X = scaler.fit_transform(train_X.astype(np.float64))
            test_X = scaler.transform(test_X.astype(np.float64))
        score = None
        if self.task == 'classification':
            self.model.fit(train_X, train_Y)
            predictions = self.model.predict(test_X)
            clf_probs = self.model.predict_proba(test_X)
            from sklearn.metrics import log_loss
            score = log_loss(test_Y, clf_probs, labels=[-1,1])
            train_acc = (sum(self.model.predict(train_X)*train_Y >0)/train_Y.shape[0])

        if self.task == 'ordinal_regression' or self.task == 'abs_ordinal_regression':
            train_Y_d = self.est.transform(train_Y.values.reshape(-1,1)).reshape(-1)
            self.model.fit(train_X, train_Y_d)
            predictions = self.est.inverse_transform(self.model.predict(test_X).reshape(-1,1)).reshape(-1)
            train_preds = self.est.inverse_transform(self.model.predict(train_X).reshape(-1,1)).reshape(-1)
            train_acc = (sum(train_preds * train_Y >0)/train_Y.shape[0])

        if self.task == 'regression':
            self.model.fit(train_X, train_Y)
            predictions = self.model.predict(test_X)
            train_acc = (sum(self.model.predict(train_X)*train_Y >0)/train_Y.shape[0])
        return predictions,train_acc,score
        
        
    def run(self, train_length=60, test_length = 1, normalize = True, sliding = False):
        """
        run backtesting
        
        args(default):
            train_length:60,
            test_length:1
            max_iterations:36
            normalize: True
            
        Returns:
            acc_df: A dataframe for the backtesting accuracy
            pred_df: A dataframe for the predictions   
        """
        results = []
        self.train_length = train_length
        self.test_length = test_length
        self.normalize = normalize
     
        for iteration in range(0, self.max_iterations):
            train_X, test_X = self.split_dataset(self.feature_df,iteration,sliding)
            train_Y, test_Y = self.split_dataset(self.target_df,iteration,sliding)
            train_T, test_T = self.split_dataset(self.date_df,iteration,sliding)
            train_X = train_X.fillna(train_X.mean())
            predictions,train_acc,score = self.train(train_X, test_X, train_Y, test_Y, train_T, test_T)
            for test_idx in range(len(predictions)):
                target_i = round(test_Y.values[test_idx],4)
                pred_i = round(predictions[test_idx],4)
                result = [target_i, pred_i, target_i*pred_i>=0, score]
                info = [train_acc, iteration, test_idx+1, self.time_horizon,self.task, self.train_length,self.test_length]
                date = [test_T.values[test_idx][0],test_T.values[test_idx][1],train_T.values[0][1],train_T.values[-1][1]]
                results.append(result + info + date)
        
        pred_df = pd.DataFrame(results,columns=['TARGET','Prediction','correct','log_loss','train_acc','train_iteration','test_iteration','time_horizon','task','train_length','test_length','target_date','feature_date','train_start','train_end'])   
        acc_df = pd.DataFrame(
        [[
            self.time_horizon,
            self.task,
            pred_df['correct'].sum()/pred_df['correct'].count(),
            sliding,
            self.max_iterations,
            train_length,
            test_length,
            pred_df["feature_date"].max(),
            pred_df["feature_date"].min(),
            pred_df["target_date"].max(),
            pred_df["target_date"].min()
        ]],
        columns=[
            "time_horizon",
            "task",
            "backtesting_accuracy",
            "sliding",
            "backtesting_iterations",
            "backtesting_training_length",
            "backtesting_test_length",
            "backtesting_end_feature_date",
            "backtesting_start_feature_date",
            "backtesting_end_target_date",
            "backtesting_start_target_date"
        ])
        self.accuracy = pred_df['correct'].sum()/pred_df['correct'].count()
        self.pred_df = pred_df
        self.acc_df = acc_df
        return pred_df,acc_df
    
    def cross_validation(self, train_length=60, sample_size=1000, normalize=True):
        self.train_length = train_length
        self.normalize = normalize
        feature_df = self.feature_df.fillna(self.feature_df.mean())
        init_idx = np.array(feature_df[:-self.time_horizon].index)
        np.random.seed(0)
        results = []
        for sample_id in range(sample_size):
            np.random.shuffle(init_idx)
            train_idx = init_idx[:train_length]
            test_idx = init_idx[train_length:]
            train_X, test_X = feature_df.loc[train_idx],feature_df.loc[test_idx]
            train_Y, test_Y = self.target_df.loc[train_idx],self.target_df.loc[test_idx]
            train_T, test_T = self.date_df.loc[train_idx],self.date_df.loc[test_idx]
            predictions,train_acc,score = self.train(train_X, test_X, train_Y, test_Y, train_T, test_T)
            
            self.test_length = test_X.shape[0]
            
            for t in range(len(predictions)):
                target_i = round(test_Y.values[t],4)
                pred_i = round(predictions[t],4)
                result = [target_i, pred_i, target_i*pred_i>=0, score]
                info = [train_acc, sample_id+1, t+1, self.time_horizon,self.task,self.train_length, self.test_length]
                date = [test_T.values[t][0],test_T.values[t][1],train_T.values[0][1],train_T.values[-1][1]]
                results.append(result + info + date)
        
        pred_df = pd.DataFrame(results,columns=['TARGET','Prediction','correct','log_loss','train_acc','train_iteration','test_iteration','time_horizon','task','train_length','test_length','target_date','feature_date','train_start','train_end'])   
        acc_df = pd.DataFrame(
        [[
            self.time_horizon,
            self.task,
            pred_df['correct'].sum()/pred_df['correct'].count(),
            sample_size
        ]],
        columns=[
            "time_horizon",
            "task",
            "backtesting_accuracy",
            "sample_size"
        ])
        self.accuracy = acc_df["backtesting_accuracy"]
        self.pred_df = pred_df
        self.acc_df = acc_df
        return pred_df, acc_df
    
    def evaluate(self):
        g = self.pred_df.set_index('train_iteration').groupby(level='train_iteration')
        output_df = g.first()[['train_start','train_end','train_length','test_length']].reset_index()
        output_df['test_start'] = g.apply(lambda x:x.feature_date.min()).values
        output_df['test_end'] = g.apply(lambda x:x.feature_date.max()).values
        output_df['Training Accuracy'] = g.apply(lambda x:x.train_acc.mean()).values
        output_df['Test Accuracy'] = g.apply(lambda x:x.correct.mean()).values
        output_df['training_test_diff'] = output_df['Test Accuracy'] - output_df['Training Accuracy']
        output_df = output_df.set_index('train_iteration')
        def f(row,metric):
            X = row.Prediction
            Y = row.TARGET
            from scipy import stats
            if metric == 'bucketed_difference':
                X_bucket = self.est.transform(X.values.reshape(-1,1)).reshape(-1)
                Y_bucket = self.est.transform(Y.values.reshape(-1,1)).reshape(-1)
                return np.abs(Y_bucket - X_bucket).mean()
            if metric == 'bucketed_difference_std':
                X_bucket = self.est.transform(X.values.reshape(-1,1)).reshape(-1)
                Y_bucket = self.est.transform(Y.values.reshape(-1,1)).reshape(-1)
                return np.abs(Y_bucket - X_bucket).std()

            if metric == 'MAE':
                from sklearn.metrics import mean_absolute_error
                return mean_absolute_error(X,Y)
            if metric == 'RMSE':
                return np.linalg.norm(X - Y) / np.sqrt(len(X))
            if metric == 'R2':
                from sklearn.metrics import r2_score
                return r2_score(Y, X)
            if metric == 'variance_of_absolute_error':
                return variance_of_absolute_error(X,Y)
            if metric == 'slope_of_absolute_error':
                return slope_of_absolute_error(X,Y)
            if metric == 'order_of_correctness':
                Y = Y > 0
                X = X > 0
                return order_of_correctness(Y,X)
            if metric == 'orderliness_of_correctitude':
                Y = Y > 0
                X = X > 0
                return orderliness_of_correctitude(Y,X)
            
            
        if self.task == 'regression' or self.task == 'ordinal_regression' or self.task == 'abs_ordinal_regression':
            for metric in ['RMSE','MAE','R2','variance_of_absolute_error','slope_of_absolute_error','order_of_correctness','orderliness_of_correctitude','bucketed_difference','bucketed_difference_std']:
                output_df[metric] = g.apply(lambda row:f(row, metric)).values
        if self.task == 'classification':
            output_df['log_loss'] = g.apply(lambda x:x.log_loss.mean()).values
            for metric in ['order_of_correctness','orderliness_of_correctitude']:
                output_df[metric] = g.apply(lambda row:f(row, metric)).values
        summary_df = output_df.mean(axis=0).to_frame().T
        summary_df['time_horizon'] = self.time_horizon
        summary_df['task'] = self.task
        return output_df,summary_df
    
    def correction_analysis(self):
        return self.pred_df.groupby('feature_date')['TARGET','Prediction','correct'].agg(['count','sum','mean','var'])