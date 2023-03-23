import pandas as pd
from pandas import DataFrame
import numpy as np


class SkatersStatsDataFrame:

    def __init__(self, file_name, previous_season_players_data=None):
        self.players_data = pd.read_csv(file_name)
        self.previous_season_players_data = previous_season_players_data

    def get_df(self) -> DataFrame:
        return self.players_data

    def get_df_for_goal_predictions(self) -> DataFrame:
        relevant_columns = ['playerId', 'name', 'position', 'games_played', 'I_F_xGoals', 'I_F_goals']
        df = self.players_data[self.players_data['situation'] == 'all']
        df = df[relevant_columns]

        # add 5on4 icetime to data frame
        icetime_5on4_data = self.players_data[self.players_data['situation'] == '5on4']
        icetime_5on4_data = icetime_5on4_data[['playerId', 'icetime']]
        icetime_5on4_data.rename(columns={'icetime': '5on4_icetime'}, inplace=True)
        df = df.merge(icetime_5on4_data, on='playerId', how='left')

        # add previous season shooting talent
        season_2021_skaters_data_all_situations = self.previous_season_players_data[self.previous_season_players_data['situation'] == 'all']
        previous_season_goal_data = season_2021_skaters_data_all_situations[['playerId', 'I_F_xGoals', 'I_F_goals']].copy()
        previous_season_goal_data['P_S_shooting_talent'] = np.where(previous_season_goal_data['I_F_xGoals'] != 0,
                                                                    (previous_season_goal_data['I_F_xGoals'] - previous_season_goal_data['I_F_goals'])
                                                                    / previous_season_goal_data['I_F_xGoals'], 0)
        previous_season_goal_data = previous_season_goal_data.drop(columns=['I_F_xGoals', 'I_F_goals'])
        df = df.merge(previous_season_goal_data, on='playerId', how='left')
        df['P_S_shooting_talent'] = df['P_S_shooting_talent'].fillna('not_applicable')

        # calculate median shooting talent
        temp_df = df[df['P_S_shooting_talent'] != 'not_applicable']
        median_shooting_talent = temp_df['P_S_shooting_talent'].astype(float).median()

        df['P_S_shooting_talent'] = np.where(df['P_S_shooting_talent'] == 'not_applicable', median_shooting_talent,
                                             df['P_S_shooting_talent'])
        df['P_S_shooting_talent'] = df['P_S_shooting_talent'].astype(float)

        return df




