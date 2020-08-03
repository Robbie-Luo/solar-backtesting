from gplearn.genetic import SymbolicRegressor
from gplearn.genetic import SymbolicClassifier
import pandas as pd
from solar.feature_selection_methods import *
data_df = pd.read_csv('TIMESERIES_all_monthly_enriched_no_dairyntel.csv').ffill()
time_horizon = 1
target_var="ZuivelNL - Dutch butter quotation"
v_feats = remove_invalid_features(data_df)
input_df = data_df[v_feats]
input_df.loc[:, "TARGET"] = -input_df[target_var].diff(-time_horizon)
input_df = input_df.dropna()
X_train = input_df.iloc[:,:-1]
Y_train = input_df.iloc[:,-1]
est_gp = SymbolicRegressor(population_size=5000,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(X_train, Y_train)
print(est_gp._program)