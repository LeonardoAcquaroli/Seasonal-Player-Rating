import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso



def get_weights(df: pd.DataFrame, columns_to_remove: list = []):
    df = df.drop(columns_to_remove, axis=1) 
    df = df.fillna(value=0) 
    
    grouped_df = df.groupby('cluster_label').sum().T

    grouped_df['total'] = grouped_df.loc[:, grouped_df.columns != 'cluster_label'].sum(axis=1)

    for col in (col for col in grouped_df.columns if col != 'cluster_label'):
        grouped_df[col] = grouped_df[col].div(grouped_df['total'])
        grouped_df[col] = grouped_df[col] / grouped_df[col].sum()

    grouped_df = grouped_df.fillna(value=0)
    grouped_df = grouped_df.drop('total', axis=1)
    grouped_df = grouped_df.to_dict()

    return grouped_df




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