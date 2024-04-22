import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso
from constants import ClusterFeaturesWeightsConstants

def get_cluster_features_weights(df: pd.DataFrame) -> dict:
    columns_to_remove = ClusterFeaturesWeightsConstants().columns_to_remove
    # Drop useless columns
    df = df.drop(columns_to_remove, axis=1)
    # Drop zero variance features
    ZERO_VAR_FEATURES = df.drop('cluster_label', axis=1).std()[df.drop('cluster_label', axis=1).std() == 0].index
    df = df.drop(ZERO_VAR_FEATURES, axis=1)
    # Fillna with 0
    df = df.fillna(value=0) 
    
    grouped_df = df.groupby('cluster_label').sum().T
    grouped_df['total'] = grouped_df.loc[:, grouped_df.columns != 'cluster_label'].sum(axis=1)

    for col in (col for col in grouped_df.columns):
        # Element-wise division
        grouped_df[col] = grouped_df[col].div(grouped_df['total'])
        minmax_scaler = MinMaxScaler()
        # MinMax scaler instead division by sum of column to have weightd from 0 to 1 that point to the importance of the feature for the cluster
        grouped_df[col] = minmax_scaler.fit_transform(grouped_df[col].values.reshape(-1, 1))
        # Fill the NaNs with the minimum weight (different from zero) in order not to have zeros and thus nullify the feature coefficient
        min_value = grouped_df[col][grouped_df[col] > 0].min()
        grouped_df[col] = grouped_df[col].replace(to_replace=0, value=min_value)

    grouped_df = grouped_df.fillna(0)
    grouped_df = grouped_df.drop('total', axis=1)
    grouped_dict = grouped_df.to_dict()
    
    return grouped_dict

def feature_selection(df: pd.DataFrame, alpha: float = 0.01):

    X = df.drop('win', axis=1)
    y = df['win']

    all_feature_names = X.columns.tolist()  

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)
    lasso_coefs = lasso.coef_

    selected_features = np.where(lasso_coefs != 0)[0]
    selected_feature_names = [all_feature_names[i] for i in selected_features]

    X_selected = X[selected_feature_names]

    return X_selected