import pandas as pd
from pandas import DataFrame
import numpy as np


class SkatersStats:

    def __init__(self, file_name, season: str, prev_season_skaters_obj=None):
        self.players_data = pd.read_csv(file_name)
        self.season = season
        self.prev_season_obj = prev_season_skaters_obj

    def get_df(self) -> DataFrame:
        return self.players_data

    def get_prev_season_obj(self):
        return self.prev_season_obj

    def get_df_for_goal_predictions(self, num_of_previous_seasons_for_goals_data=2) -> DataFrame:
        relevant_columns = ['playerId', 'name', 'position', 'games_played', 'I_F_xGoals', 'I_F_goals']
        df = self.players_data[self.players_data['situation'] == 'all']
        df = df[relevant_columns]

        # add 5on4 icetime to data frame
        icetime_5on4_data = self.players_data[self.players_data['situation'] == '5on4']
        icetime_5on4_data = icetime_5on4_data[['playerId', 'icetime']]
        icetime_5on4_data.rename(columns={'icetime': '5on4_icetime'}, inplace=True)
        df = df.merge(icetime_5on4_data, on='playerId', how='left')

        df = self.add_prev_season_shooting_talent(df, number_of_seasons=num_of_previous_seasons_for_goals_data)

        return df

    def add_prev_season_shooting_talent(self, df, number_of_seasons=1):
        current = None

        for i in range(1, number_of_seasons + 1):
            if current is None:
                current = self.get_prev_season_obj()
            else:
                current = current.get_prev_season_obj()

            prev_season_df = current.get_df()
            prev_season_skaters_all_stats = prev_season_df[
                prev_season_df['situation'] == 'all']
            prev_season_skaters_goal_stats = prev_season_skaters_all_stats[
                ['playerId', 'I_F_xGoals', 'I_F_goals']].copy()

            prev_season_skaters_goal_stats[f'{i}_season_ago_shooting_talent'] = np.where(
                prev_season_skaters_goal_stats['I_F_xGoals'] != 0,
                (prev_season_skaters_goal_stats['I_F_xGoals'] -
                 prev_season_skaters_goal_stats['I_F_goals'])
                / prev_season_skaters_goal_stats['I_F_xGoals'], 0)

            prev_season_skaters_goal_stats = prev_season_skaters_goal_stats.drop(columns=['I_F_xGoals'])

            prev_season_skaters_goal_stats.rename(columns={'I_F_goals': f'{i}_season_ago_goals'}, inplace=True)

            df = df.merge(prev_season_skaters_goal_stats, on='playerId', how='left')
            df[f'{i}_season_ago_shooting_talent'] = df[f'{i}_season_ago_shooting_talent'].fillna(
                'not_applicable')
            # calculate median shooting talent
            temp_df = df[df[f'{i}_season_ago_shooting_talent'] != 'not_applicable']
            median_shooting_talent = temp_df[f'{i}_season_ago_shooting_talent'].astype(float).median()
            df[f'{i}_season_ago_shooting_talent'] = np.where(
                df[f'{i}_season_ago_shooting_talent'] == 'not_applicable', median_shooting_talent,
                df[f'{i}_season_ago_shooting_talent'])
            df[f'{i}_season_ago_shooting_talent'] = df[f'{i}_season_ago_shooting_talent'].astype(float)

            df[f'{i}_season_ago_goals'] = df[f'{i}_season_ago_goals'].fillna(0.0)

        return df
