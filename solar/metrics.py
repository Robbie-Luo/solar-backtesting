'''
Extended model evaluation metrics, complementing the existing scikit-learn metrics.
Author: Yalun Zheng
'''
import numpy as np

def order_of_correctness(y_true, y_pred):
    '''
    This function is designed to measure whether a model makes correct predictions
    in a specific order. Consider the example below:
    
        y_true = np.array([1,1,0,0])
        y_pred_1 = np.array([1,1,1,1])
        y_pred_2 = np.array([0,0,0,0])
        y_pred_3 = np.array([1,0,0,1])
        y_pred_4 = np.array([0,1,1,0])
        
    We assume that the 4 sets of predictions come from 4 models. Then the 4 models 
    have the same accuracy: 50%. However, under the context of Timeseries analysis,
    we believe that these 4 models behave quite differently. If the predictions are
    timely ordered, then we can observe that the 1st model predicts correctly at the
    beginning of the testing period and goes wrong later (degrading performance),
    and the 2nd model does the opposite (upgrading performance). The 3rd & 4th models
    act similarly without showing any timely skewed performance.
    
    With these in mind, this function will return a value in the range of [-1, 1]. For
    upgrading performance, the return value should be close to 1, and -1 for degrading
    performance. The less ordered the predictions are, the lower the absolute value of
    the return value. Consider the example below:
    
        order_of_correctness(y_true, y_pred_1),\
        order_of_correctness(y_true, y_pred_2),\
        order_of_correctness(y_true, y_pred_3),\
        order_of_correctness(y_true, y_pred_4)
        
    The output is:
    
        (-1.0, 1.0, 0.0, 0.0)
        
    Please only use this function for evaluating classification models.
        
    Arguments:
    ========
    y_true: numpy array with actual class labels.
    y_pred: numpy array with predicted class labels.
    
    Returns:
    ========
    Real number value indicating order of correctness, range [-1, 1].
    '''
    from scipy.stats import kendalltau as k
    y_crct = y_true == y_pred
    return k(y_crct, np.sort(y_crct))[0]

def orderliness_of_correctitude(y_true, y_pred):
    '''
    This is the twin function of order_of_correctness. This function only shows the
    degree of being ordered, i.e., orderliness. It will return a value in range [0, 1].
    So if the correctness of the predictions are more timely ordered (with no regard
    to up/degrading), a larger value will be returned. Consider the similar example:
    
        y_true = np.array([1,1,0,0])
        y_pred_1 = np.array([1,1,1,1])
        y_pred_2 = np.array([0,0,0,0])
        y_pred_3 = np.array([1,0,0,1])
        y_pred_4 = np.array([0,1,1,0])
        
        orderliness_of_correctitude(y_true, y_pred_1),\
        orderliness_of_correctitude(y_true, y_pred_2),\
        orderliness_of_correctitude(y_true, y_pred_3),\
        orderliness_of_correctitude(y_true, y_pred_4)
        
    The output is:
    
        (1.0, 1.0, 0.0, 0.0)
        
    Please only use this function for evaluating classification models.
        
    Arguments:
    ========
    y_true: numpy array with actual class labels.
    y_pred: numpy array with predicted class labels.
    
    Returns:
    ========
    Real number value indicating order of correctness, range [0, 1].
    '''
    from scipy.stats import kendalltau as k
    y_crct = y_true == y_pred
    return np.max([
        k(y_crct, np.sort(y_crct))[0], 
        k(y_crct, np.sort(y_crct)[::-1])[0]
    ])

def variance_of_absolute_error(y_true, y_pred):
    '''
    This functions measures the variance of absolute error of predictions.
    Since standard deviation is more meaningful than variance in terms of
    error analysis, we use std() instead of var().
    
    Please only use this function for evaluating regression models.
        
    Arguments:
    ========
    y_true: numpy array with actual regressand values.
    y_pred: numpy array with predicted regressand values.
    
    Returns:
    ========
    Real number value indicating the variance, range [0, +inf).
    '''
    return np.abs(y_true - y_pred).std()

def slope_of_absolute_error(y_true, y_pred):
    '''
    This function is designed to estimate the slope of absolute error of predictions
    along time. By fitting a straight line on time index of 
    
    Arguments:
    ========
    y_true: numpy array with actual regressand values.
    y_pred: numpy array with predicted regressand values.
    
    Returns:
    ========
    Real number value represent the slope, range (-inf, +inf).
    '''
    from scipy.stats import linregress
    return linregress(
        np.arange(0, y_true.shape[0]),
        np.abs(y_true - y_pred)
    )[0]