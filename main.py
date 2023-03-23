import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skaters_stats_df import SkatersStatsDataFrame

skaters_df_2018 = SkatersStatsDataFrame('data/skaters_2018.csv')
skaters_df_2019 = SkatersStatsDataFrame('data/skaters_2019.csv', previous_season_players_data=skaters_df_2018.get_df())
skaters_df_2020 = SkatersStatsDataFrame('data/skaters_2020.csv', previous_season_players_data=skaters_df_2019.get_df())
skaters_df_2021 = SkatersStatsDataFrame('data/skaters_2021.csv', previous_season_players_data=skaters_df_2020.get_df())
skaters_df_2022 = SkatersStatsDataFrame('data/skaters_2022.csv', previous_season_players_data=skaters_df_2021.get_df())

df_train = pd.concat([skaters_df_2019.get_df_for_goal_predictions(), skaters_df_2020.get_df_for_goal_predictions(), skaters_df_2021.get_df_for_goal_predictions()], ignore_index=True)
df_eval = skaters_df_2022.get_df_for_goal_predictions()
y_train = df_train.pop('I_F_goals')
y_eval = df_eval.pop('I_F_goals')
print(y_eval)

