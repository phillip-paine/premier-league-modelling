import pandas as pd
from typing import Dict


def create_across_seasons_coefs(map_season_coefs: Dict[str, pd.DataFrame]):
    df_team_coefs = pd.DataFrame()
    for season, df in map_season_coefs.items():
        df.rename(columns={'att': f'att_{season}', 'def': f'def_{season}'}, inplace=True)
        df.drop(columns=['index', 'team_id'], inplace=True)
        if df_team_coefs.empty:
            df_team_coefs = df
        else:
            df_team_coefs = pd.merge(df_team_coefs, df, on=['team_name'], how='outer')
    df_team_coefs['att'] = df_team_coefs[[c for c in df_team_coefs.columns if c.startswith('att')]].mean(axis=1)
    df_team_coefs['def'] = df_team_coefs[[c for c in df_team_coefs.columns if c.startswith('def')]].mean(axis=1)
    df_team_coefs = df_team_coefs[['team_name', 'att', 'def']]
    return df_team_coefs


def create_new_season_coefs(history_season_coefs: pd.DataFrame, new_season_coefs: pd.DataFrame, current_gameweek: int) -> pd.DataFrame:
    modelling_coefs_df = pd.merge(history_season_coefs[['team_name', 'att', 'def']], new_season_coefs[['team_name', 'att', 'def']], on=['team_name'], suffixes=('_history', '_current'), how='outer')
    modelling_coefs_df['att'] = modelling_coefs_df[[c for c in modelling_coefs_df.columns if c.startswith('att')]].mean(axis=1)
    modelling_coefs_df['def'] = modelling_coefs_df[[c for c in modelling_coefs_df.columns if c.startswith('def')]].mean(axis=1)
    return modelling_coefs_df[['team_name', 'att', 'def']]
