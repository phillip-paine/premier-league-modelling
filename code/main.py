from classes import DataPreparation, Modelling, Results
import os
import numpy as np

from utils import create_across_seasons_coefs, create_new_season_coefs

DIR = "~/Documents/Code/premier-league-modelling"


def main():
    seasons = ['2021', '2122', '2223']
    # seasons = ['2122']
    dataprep = DataPreparation(seasons=seasons)
    dataprep.store_locally(overwrite=True)
    data = dataprep.return_data()
    team_coefs_dict = {}
    home_adv_dict = {}
    # These are the full years:
    for season in seasons:

        data_season = data[data['season'] == season]
        pn_model = Modelling(df=data_season,
                             code_model_filepath=os.path.join(DIR, "model_files", "poisson_normal_hierarchical_basic.stan"),
                             existing_trained_model_path=os.path.join(DIR, "model_files",
                                                                      f"trained_pois_normal_hierarchical_basic_{season}.pkl"))
        pn_model.fit_model(m=500)
        team_coefs_df = pn_model.create_team_index_coef_dataframe()
        team_coefs_dict[season] = team_coefs_df
        home_adv = pn_model.get_posterior_samples_mean()['mean_home_samples']
        home_adv_dict[season] = home_adv

    # Coefficients of the completed seasons:
    model_team_coefs = create_across_seasons_coefs(team_coefs_dict)

    # Current (partially completed) season:
    # TODO some sort of try if the data is empty and we don't need to try and fit to result yet etc.
    new_season = ['2324']
    dataprep = DataPreparation(seasons=new_season)
    dataprep.store_locally(overwrite=True)
    new_data = dataprep.return_data()
    fixture_set = new_data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HomeTeamInt', 'AwayTeamInt']]
    max_date = fixture_set['Date'].max()  # this is a string date
    # max_date = "2023-08-01"  # can set to this to get a before season prediction
    max_date = "2023-12-31"  # test roughly half-way through the last season

    # TODO Fit the fixtures completed so far in the current season:
    data_current_season = fixture_set[fixture_set['Date'] <= max_date]
    # TODO we need to add the max_date to the new_season pkl file maybe? so we can get it back later if we want it.
    pn_model = Modelling(df=data_current_season,
                         code_model_filepath=os.path.join(DIR, "model_files", "poisson_normal_hierarchical_basic.stan"),
                         existing_trained_model_path=os.path.join(DIR, "model_files",
                                                                  f"trained_pois_normal_hierarchical_basic_{new_season[0]}.pkl"))
    pn_model.fit_model(m=500)
    team_coefs_df_new_season = pn_model.create_team_index_coef_dataframe()
    home_adv_dict[new_season[0]] = pn_model.get_posterior_samples_mean()['mean_home_samples']

    # TODO Create the model coefficients (att, def and home_adv)
    average_model_coefs_df = create_new_season_coefs(model_team_coefs, team_coefs_df_new_season, dataprep.gameweek)
    model_home_adv = np.mean(list(home_adv_dict.values()))

    # TODO add the current data + predict future fixtures + create completed fixtures + predict fixtures table
    model_preds = Results(average_model_coefs_df, fixture_set, model_home_adv, max_date)
    # prediction table:
    # mean_prediction_table = model_preds.table_mean_goals
    # model_preds.display_table(mean_prediction_table,
    #                           os.path.join(DIR, "output", f"league_table_plot_{new_season[0]}.png"))
    # Monte Carlo Probability Table:
    mc_seasons_dfs = model_preds.monte_carlo_season_simulation(n_mc=500)
    map_probs_df = model_preds.final_table_team_probabilities(mc_seasons_dfs)
    return


if __name__ == '__main__':
    main()
