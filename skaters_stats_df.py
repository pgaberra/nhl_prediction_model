import pandas as pd
from pandas import DataFrame
import numpy as np


class SkatersStats:

    def __init__(self, file_name, season: str):
        """
        Initialize a SkatersStats object with player statistics from a CSV file.

        :param file_name: The path to the CSV file containing player statistics
        :param season: A string representing the season of the player statistics
        """
        try:
            self.players_data = pd.read_csv(file_name)
        except FileNotFoundError:
            print(f"Error: The file {file_name} could not be found.")
            exit()
        except Exception as e:
            print(f"Error: An unexpected error occurred while reading the file {file_name}: {e}")
            exit()
        self.season = season

    def get_df(self) -> DataFrame:
        """
        Return the entire DataFrame containing all player statistics.

        :return: A DataFrame containing all player statistics
        """
        return self.players_data

    def get_df_for_goal_predictions(self) -> DataFrame:
        """
        Get a DataFrame with relevant columns for goal predictions.

        :return: A DataFrame containing relevant columns for goal predictions
        """
        relevant_columns = ['playerId', 'name', 'position', 'icetime', 'games_played', 'I_F_shotsOnGoal', 'I_F_flurryAdjustedxGoals', 'I_F_goals']
        df = self.players_data[self.players_data['situation'] == 'all']
        df = df[relevant_columns]

        # add 5on4 icetime to data frame
        icetime_5on4_data = self.players_data[self.players_data['situation'] == '5on4']
        icetime_5on4_data = icetime_5on4_data[['playerId', 'icetime']]
        icetime_5on4_data.rename(columns={'icetime': '5on4_icetime'}, inplace=True)
        df = df.merge(icetime_5on4_data, on='playerId', how='left')

        return df

    def get_df_for_assist_predictions(self) -> DataFrame:
        """
        Get a DataFrame with relevant columns for assist predictions.

        :return: A DataFrame containing relevant columns for assist predictions
        """
        relevant_columns = ['playerId', 'name', 'position', 'icetime', 'games_played', 'OnIce_F_flurryAdjustedxGoals', 'OnIce_F_shotsOnGoal', 'onIce_corsiPercentage', 'I_F_primaryAssists', 'I_F_secondaryAssists']
        df = self.players_data[self.players_data['situation'] == 'all']
        df = df[relevant_columns]

        icetime_5on4_data = self.players_data[self.players_data['situation'] == '5on4']
        icetime_5on4_data = icetime_5on4_data[['playerId', 'icetime']]
        icetime_5on4_data.rename(columns={'icetime': '5on4_icetime'}, inplace=True)
        df = df.merge(icetime_5on4_data, on='playerId', how='left')

        df['I_F_assists'] = df['I_F_primaryAssists'] + df['I_F_secondaryAssists']

        df = df.drop(['I_F_primaryAssists', 'I_F_secondaryAssists'], axis=1)

        return df
