import pandas as pd
import numpy as np
import os
from typing import List, Any, Dict, Optional
import pickle
import stan
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce, partial

DIR = "~/Documents/Code/premier-league-modelling"


class DataPreparation:
    def __init__(self, seasons=None):
        if seasons is None:
            self.seasons = ['2021', '2122', '2223']
        else:
            self.seasons = seasons
        self.filepath = os.path.join(DIR, "data")
        self.df = self._get_data()
        self.team_map = self._create_team_map()
        self._prepare_data()

    def _get_data(self):
        if os.path.exists(self.filepath):
            self.df = pd.read_csv(self.filepath)
        else:
            self.df = self._retrieve_data()
        return self.df

    def _retrieve_data(self):
        season_data_urls = [f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv' for season in self.seasons]
        df_store = pd.DataFrame()
        for url in season_data_urls:
            try:
                season = url.split('/')[-2]
                df = pd.read_csv(url)
                df = df.iloc[:, :24]  # if we go back further perhaps the columns wont match anymore
                df['season'] = season
                if df_store.empty:
                    df_store = df
                else:
                    df_store = pd.concat([df_store, df])
            except Exception as e:
                print(f'Error found {e}')
        return df_store

    def _create_team_map(self):
        team_map: Dict[str, Dict[str, int]] = {}
        for season in self.seasons:

            df_teams = self.df[self.df['season'] == season][['HomeTeam']].drop_duplicates()
            df_teams.sort_values(by=['HomeTeam'], ascending=True, inplace=True)
            df_teams.reset_index(inplace=True)
            season_team_map: Dict[str, int] = {}
            for index, row in df_teams.iterrows():
                season_team_map.update({row['HomeTeam']: index+1})
            team_map.update({season: season_team_map})
        return team_map

    def _prepare_data(self):
        self.df['date_object'] = pd.to_datetime(self.df['Date'], format="%d/%m/%Y")
        self.df['hour'] = self.df.apply(lambda row: row['Time'].split(':')[0], axis=1)
        self.df['month'] = self.df['date_object'].dt.month
        self.df['goal_diff'] = abs(self.df['FTHG'] - self.df['FTAG'])
        self.df['goals_total'] = self.df['FTHG'] + self.df['FTAG']
        self.df['home_win'] = np.where(self.df['FTHG'] > self.df['FTAG'], 1, 0)
        home_ftr_dict = {'H': 3, 'D': 1, 'A': 0}
        self.df['home_pts'] = self.df.apply(lambda row: home_ftr_dict[row['FTR']], axis=1)
        away_ftr_dict = {'A': 3, 'D': 1, 'H': 0}
        self.df['away_pts'] = self.df.apply(lambda row: away_ftr_dict[row['FTR']], axis=1)
        self.df['draws'] = np.where(self.df['FTR'] == "D", 1, 0)
        self.df['match'] = 1
        self.df['HomeTeamInt'] = self.df.apply(lambda row: self.team_map.get(row['season']).get(row['HomeTeam']), axis=1)
        self.df['AwayTeamInt'] = self.df.apply(lambda row: self.team_map.get(row['season']).get(row['AwayTeam']), axis=1)

    def store_locally(self, overwrite: bool):
        if os.path.exists(self.filepath) and not overwrite:
            print('Path already exists')
        else:
            self.df.to_csv(os.path.join(DIR, "data", f"match_results_{'_'.join(self.seasons)}"))
            print('CSV written')
        return None

    def return_data(self) -> pd.DataFrame:
        return self.df

    def return_team_map(self):
        return self.team_map


class Modelling:
    def __init__(self, df: pd.DataFrame, code_model_filepath: str, existing_trained_model_path: Optional[str] = None, init_values: Optional[List[float]] = None):
        self.model_code = None
        self.model = None
        self.samples = None
        self.posterior_samples = None
        self.df = df
        self.model_data = self._create_model_data(df)
        self.init_values = init_values
        self.code_model_filepath = os.path.expanduser(code_model_filepath)
        self.trained_model_filepath = os.path.expanduser(existing_trained_model_path)
        self.model = self.fetch_trained_model(self.trained_model_filepath)
        self.read_model_code()

    @staticmethod
    def _create_model_data(df: pd.DataFrame):
        model_data: Dict[str, List[Any]] = {
            "y1": df['FTHG'].tolist(),
            "y2": df['FTAG'].tolist(),
            "hometeam": df['HomeTeamInt'].tolist(),
            "awayteam": df['AwayTeamInt'].tolist(),
            "nteams": 20,
            "ngames": len(df.index),
        }
        return model_data

    def fetch_trained_model(self, trained_model_path: Optional[str]):
        if trained_model_path:
            try:
                with open(trained_model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except FileNotFoundError as e:
                print(e)
                return None
            return self.model
        else:
            return None

    def read_model_code(self):
        try:
            with open(self.code_model_filepath) as file:
                self.model_code = file.read()
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)

    def fit_model(self, m: int) -> None:
        if self.model_code:
            if self.model:
                # Burn-in
                fit = self.model.sample(num_chains=1, num_samples=m, num_warmup=500)

                df = fit.to_frame()
                self.samples = df
                self.posterior_samples = self.posterior_parameter_samples()
            else:
                self.model = stan.build(program_code=self.model_code, data=self.model_data)

                # Burn-in
                fit = self.model.sample(num_chains=1, num_samples=m, num_warmup=500)

                # store fitted model:
                with open(self.trained_model_filepath, 'wb') as f:
                    pickle.dump(self.model, f)

                df = fit.to_frame()
                self.samples = df
                self.posterior_samples = self.posterior_parameter_samples()

        else:
            raise ValueError("No model code loaded - please check that the bugs file is in the correct location")
        return None

    def posterior_parameter_samples(self) -> Dict[str, Any]:
        # Print summary statistics
        coef_samples = self.samples[[c for c in self.samples.columns if c.startswith('att.') or c.startswith('def.')]]  # e.g. att.1, att.2, ...
        home_samples = self.samples[['home']]
        return {"team_coef_samples": coef_samples, "home_samples": home_samples}

    def get_posterior_samples_mean(self):
        return {"mean_home_samples": self.posterior_samples['home_samples'].mean(),
                'mean_team_coef_samples': self.posterior_samples['team_coef_samples'].mean().to_frame().transpose()}

    def create_team_index_coef_dataframe(self):
        df = self.get_posterior_samples_mean()['mean_team_coef_samples']
        # Attack
        df_att = df[[c for c in df.columns if c.startswith('att')]]
        # Melt the DataFrame
        df_att = pd.melt(df_att, var_name='index', value_name='att')

        # Extract the numeric part from the 'index' column
        df_att['team_index'] = df_att['index'].str.extract('(\d+)').astype(int)

        # Sort by 'index' to ensure proper ordering
        df_att = df_att.sort_values(by='team_index').reset_index(drop=True)

        # Defense:
        df_def = df[[c for c in df.columns if c.startswith('def')]]
        # Melt the DataFrame
        df_def = pd.melt(df_def, var_name='index', value_name='def')

        # Extract the numeric part from the 'index' column
        df_def['team_index'] = df_def['index'].str.extract('(\d+)').astype(int)

        # Sort by 'index' to ensure proper ordering
        df_def = df_def.sort_values(by='team_index').reset_index(drop=True)

        df = pd.merge(df_att[['team_index', 'att']], df_def[['team_index', 'def']], on=['team_index'], how='inner').reset_index()

        df = pd.merge(df, self.df[['HomeTeamInt', 'HomeTeam']].drop_duplicates(), left_on='team_index', right_on='HomeTeamInt')
        df.rename(columns={'HomeTeam': 'team_name', 'HomeTeamInt': 'team_id'}, inplace=True)
        df.drop(columns=['team_index'], inplace=True)

        return df

    def retrieve_diagnostics(self):
        return self.model


class Results:
    def __init__(self, team_coefs, fixtures, home_adv):
        self.df_goals_for_against = pd.DataFrame()
        self.complete_table_df = None
        self.df_mean_goals = pd.DataFrame()
        self.df_mean_results_set = pd.DataFrame()
        self.table_mean_goals = pd.DataFrame()

        self.home_adv = home_adv
        self.df_coef = team_coefs
        self._team_normalised_goals_for_and_against()
        self.df_fixture = fixtures
        self.df = self._merge_team_coefs(self.df_fixture, self.df_coef)

        self._result_prediction_exact_expectation()
        self.df_mean_results_set = self.create_results_set(df_prediction=self.df_mean_goals, prediction_set=True)
        self.table_mean_goals = self.create_table(self.df_mean_results_set)

    def _team_normalised_goals_for_and_against(self):
        """Normalise the coefs to find the expected goals scored and conceded against a 'neutral' opponent"""
        self.df_goals_for_against = self.df_coef[['team_name', 'att', 'def']]
        att_mean = self.df_goals_for_against['att'].mean()
        def_mean = self.df_goals_for_against['def'].mean()

        # Goals for and against the mean home and away att+def coefficients:
        self.df_goals_for_against['For'] = np.exp(self.df_goals_for_against['att'] + def_mean).round(1)
        self.df_goals_for_against['Against'] = np.exp(self.df_goals_for_against['def'] + att_mean).round(1)

        self.df_goals_for_against.rename(columns={'team_name': 'Team'}, inplace=True)
        self.df_goals_for_against = self.df_goals_for_against[['Team', 'For', 'Against']]
        return None

    @staticmethod
    def _merge_team_coefs(fixture, coefs):
        # Home team coefs:
        df = pd.merge(fixture, coefs, left_on=['HomeTeam'], right_on=['team_name'], how='left')
        df.rename(columns={'att': 'hometeam_att', 'def': 'hometeam_def'}, inplace=True)
        promoted_home_att_coef = df['hometeam_att'].min()
        promoted_home_def_coef = df['hometeam_def'].min()
        df['hometeam_att'].fillna(value=promoted_home_att_coef, inplace=True)
        df['hometeam_def'].fillna(value=promoted_home_def_coef, inplace=True)
        df.drop(columns=['team_name'], inplace=True)
        # Away team coefs:
        df = pd.merge(df, coefs, left_on=['AwayTeam'], right_on=['team_name'], how='left')
        df.rename(columns={'att': 'awayteam_att', 'def': 'awayteam_def'}, inplace=True)
        df.drop(columns=['team_name'], inplace=True)
        promoted_away_att_coef = df['awayteam_att'].min()
        promoted_away_def_coef = df['awayteam_def'].min()
        df['awayteam_att'].fillna(value=promoted_away_att_coef, inplace=True)
        df['awayteam_def'].fillna(value=promoted_away_def_coef, inplace=True)

        return df

    def create_results_set(self, df_prediction: pd.DataFrame = pd.DataFrame(), prediction_set: bool = True) -> pd.DataFrame:
        results_col = ""
        if prediction_set:
            results_col = "predicted_result"
        else:
            results_col = 'FTR'
        if df_prediction.empty:
            df_prediction = self.df
        df_prediction['home_win'] = np.where(df_prediction[results_col] == "H", 1, 0)
        df_prediction['home_draw'] = np.where(df_prediction[results_col] == "D", 1, 0)
        df_prediction['home_loss'] = np.where(df_prediction[results_col] == "A", 1, 0)
        df_prediction['home_points'] = np.select([df_prediction['home_win'] == 1, df_prediction['home_loss'] == 1],
                                           [3, 0], default=1)

        df_prediction['away_win'] = np.where(df_prediction[results_col] == "A", 1, 0)
        df_prediction['away_draw'] = np.where(df_prediction[results_col] == "D", 1, 0)
        df_prediction['away_loss'] = np.where(df_prediction[results_col] == "H", 1, 0)
        df_prediction['away_points'] = np.select([df_prediction['away_win'] == 1, df_prediction['away_loss'] == 1],
                                           [3, 0], default=1)

        return df_prediction

    def _result_prediction_exact_expectation(self):
        self.df_mean_goals = self.df.copy(deep=True)
        self.df_mean_goals['hometeam_lambda'] = np.exp(self.home_adv + self.df_mean_goals['hometeam_att'] + self.df_mean_goals['awayteam_def']).round(0).astype(int)
        self.df_mean_goals['awayteam_lambda'] = np.exp(self.df_mean_goals['awayteam_att'] + self.df_mean_goals['hometeam_def']).round(0).astype(int)
        self.df_mean_goals['predicted_result'] = "D"
        self.df_mean_goals['predicted_result'] = np.select([self.df_mean_goals['hometeam_lambda'] > self.df_mean_goals['awayteam_lambda'], self.df_mean_goals['hometeam_lambda'] < self.df_mean_goals['awayteam_lambda']],
                                                ["H", "A"], default=self.df_mean_goals['predicted_result'])
        return None

    def result_prediction_model_simulation(self):
        # draw a score from the distribution instead of using expectation
        # e.g. a step in a monte carlo simulation
        df = self.df.copy(deep=True)
        df['hometeam_goals_draw'] = np.random.poisson(np.exp(self.home_adv + df['hometeam_att'] + df['awayteam_def'])).round(0).astype(int)
        df['awayteam_goals_draw'] = np.random.poisson(np.exp(df['awayteam_att'] + df['hometeam_def'])).round(0).astype(int)
        df['predicted_result'] = "D"
        df['predicted_result'] = np.select([df['hometeam_goals_draw'] > df['awayteam_goals_draw'], df['hometeam_goals_draw'] < df['awayteam_goals_draw']],
                                                ["H", "A"], default=df['predicted_result'])
        return df

    def monte_carlo_season_simulation(self, n_mc: int = 100) -> List[pd.DataFrame]:
        """Runs the result_predition_model_simulation func a bunch of times to get e.g probability of position
        finishes for each team"""
        simulated_seasons_list = []
        for i in range(n_mc):
            predicted_results = self.result_prediction_model_simulation()
            df_results = self.create_results_set(predicted_results)
            predicted_table = self.create_table(df_results)
            predicted_table = predicted_table[['Team', 'points']]
            predicted_table['finish'] = range(1, 21, 1)
            predicted_table.rename(columns={'points': f'points_{str(i)}', 'finish': f'finish_{str(i)}'}, inplace=True)
            simulated_seasons_list.append(predicted_table)

        return simulated_seasons_list

    @staticmethod
    def _merge_season_dfs(df1, df2, merge_cols):
        return pd.merge(df1, df2, on=merge_cols)

    def final_table_team_probabilities(self, season_simulation_list: List[pd.DataFrame]):
        """Returns a team x position table where the elements are the prob. the team finishes in that position from a
        monte carlo simulation"""
        # df = reduce(self._merge_season_dfs, season_simulation_list)
        # Create a partial function with the 'column' argument fixed
        merge_func_with_column = partial(self._merge_season_dfs, merge_cols='Team')

        # Use reduce to merge all DataFrames in the list
        df = reduce(lambda x, y: merge_func_with_column(x, y), season_simulation_list)

        # create a position finish probabilites table:
        finish_cols = [c for c in df if c.startswith('finish')]
        df_finish_probs = self._create_multiple_season_percentages(df, finish_cols, 'finish')
        df_finish_probs = df_finish_probs.merge(self.table_mean_goals[['Team', 'points']], on=['Team']).sort_values(by=['points'], ascending=False).rename(columns={'points': 'mean_points'})
        # create a points finish probabilities table:
        points_cols = [c for c in df if c.startswith('points')]
        df_pts_probs = self._create_multiple_season_percentages(df, points_cols, 'points')
        df_pts_probs = df_pts_probs.merge(self.table_mean_goals[['Team', 'points']], on=['Team']).sort_values(by=['points'], ascending=False).rename(columns={'points': 'mean_points'})
        return {'finish_probabilities': df_finish_probs, 'points_probabilities': df_pts_probs}

    @staticmethod
    def _create_multiple_season_percentages(df: pd.DataFrame, data_columns, value_column):
        df = pd.melt(df, id_vars='Team', value_vars=data_columns, var_name='season', value_name=value_column)
        df = df.groupby(['Team', value_column]).size().unstack(fill_value=0)
        df = df.div(df.sum(axis=1), axis=0) * 100
        df = df.astype(int)
        df = df.reset_index()
        return df

    def create_table(self, df_results: pd.DataFrame, end_date: Optional[str] = None) -> pd.DataFrame:
        results_df = df_results.copy()
        if end_date:
            results_df = results_df[results_df['date'] <= end_date]
        results_df['home_games'] = 1
        results_df['away_games'] = 1

        home_table_df = results_df.groupby(['HomeTeam'])[['home_games', 'home_win', 'home_draw', 'home_loss', 'home_points']].sum().reset_index()

        # If a team hasn't played a home game yet then add them to the table with 0
        remaining_team_df = self.df[~self.df['HomeTeam'].isin(home_table_df['HomeTeam'])][['HomeTeam']].drop_duplicates()
        home_table_df = home_table_df.merge(remaining_team_df, on=['HomeTeam'], how='outer')
        home_table_df.rename(columns={'HomeTeam': 'Team'}, inplace=True)
        home_table_df = home_table_df.fillna(value=0)

        away_table_df = results_df.groupby(['AwayTeam'])[['away_games', 'away_win', 'away_draw', 'away_loss', 'away_points']].sum().reset_index()
        # If a team hasn't played away yet then add them to the tale with 0 pld, pts etc.
        remaining_team_df = self.df[~self.df['AwayTeam'].isin(away_table_df['AwayTeam'])][['AwayTeam']].drop_duplicates()
        away_table_df = away_table_df.merge(remaining_team_df, on=['AwayTeam'], how='outer')
        away_table_df.rename(columns={'AwayTeam': 'Team'}, inplace=True)
        away_table_df = away_table_df.fillna(value=0)

        table_df = pd.merge(home_table_df, away_table_df, on=['Team'], how='outer')

        for event in ['games', 'win', 'draw', 'loss', 'points']:
            table_df[event] = table_df[[c for c in table_df.columns if c.endswith(event)]].sum(axis=1)
        table_df.sort_values(by=['points'], ascending=False, inplace=True)

        return table_df

    @staticmethod
    def display_single_table(display_df: pd.DataFrame, save_filepath):
        # Put the table in reverse order first:
        display_df = display_df.sort_values(by=['points'], ascending=True)
        fig = plt.figure(figsize=(7, 10), dpi=300)
        ax = plt.subplot()

        ncols = 6
        nrows = display_df.shape[0]

        ax.set_xlim(0, ncols + 1)
        ax.set_ylim(0, nrows)

        positions = [0.25, 2.5, 3.5, 4.5, 5.5, 6.5]
        columns = ['Team', 'games', 'win', 'draw', 'loss', 'points']

        # Add table's main text
        for i in range(nrows):
            for j, column in enumerate(columns):
                if j == 0:
                    ha = 'left'
                else:
                    ha = 'center'
                ax.annotate(
                    xy=(positions[j], i + .5),
                    text=display_df[column].iloc[i],
                    ha=ha,
                    va='center'
                )

        # Add column names
        column_names = ['Team', 'P', 'W', 'D', 'L', 'Pts']
        for index, c in enumerate(column_names):
            if index == 0:
                ha = 'left'
            else:
                ha = 'center'
            ax.annotate(
                xy=(positions[index], nrows),
                text=column_names[index],
                ha=ha,
                va='bottom',
                weight='bold'
            )
        ax.fill_between(
            x=[0,2],
            y1=nrows,
            y2=0,
            color='lightgrey',
            alpha=0.5,
            ec='None'
        )
        ax.set_axis_off()
        plt.savefig(
            os.path.expanduser(save_filepath),
            dpi=300,
            transparent=False,
            bbox_inches='tight'
        )
        return None

    @staticmethod
    def display_monte_carlo_table(df: pd.DataFrame):

        return None

    @staticmethod
    def display_probabilities_table(df_to_display: pd.DataFrame):
        return None


class Analysis:
    def __init__(self, match_results: pd.DataFrame, predicted_results: pd.DataFrame):
        self.predicted_df = predicted_results
        self.true_df = match_results
        self.df = self.create_comparison_fixtures()

    @staticmethod
    def create_comparison_table(self):
        return pd.merge(self.predicted_df, self.true_df, on=['HomeTeam', 'AwayTeam'], how='inner')

    def create_comparison_fixtures(self):
        df = pd.DataFrame()
        return df
