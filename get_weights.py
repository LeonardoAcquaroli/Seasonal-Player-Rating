import pandas as pd

def get_weights(df: pd.DataFrame):
    df = df.drop(['competition', 'team', 'full_name', 'stat_type', 'side', 'created_at', 'updated_at', 'player_id', 'team_id', 'game_id', 'competition_id', 'season'], axis=1) 
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