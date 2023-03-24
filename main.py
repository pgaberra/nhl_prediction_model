import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from skaters_stats_df import SkatersStats

skaters_df_2016 = SkatersStats('data/skaters_2016.csv', '2016')
skaters_df_2017 = SkatersStats('data/skaters_2017.csv', '2017', prev_season_skaters_obj=skaters_df_2016)
skaters_df_2018 = SkatersStats('data/skaters_2018.csv', '2018', prev_season_skaters_obj=skaters_df_2017)
skaters_df_2019 = SkatersStats('data/skaters_2019.csv', '2019', prev_season_skaters_obj=skaters_df_2018)
skaters_df_2020 = SkatersStats('data/skaters_2020.csv', '2020', prev_season_skaters_obj=skaters_df_2019)
skaters_df_2021 = SkatersStats('data/skaters_2021.csv', '2021', prev_season_skaters_obj=skaters_df_2020)
skaters_df_2022 = SkatersStats('data/skaters_2022.csv', '2022', prev_season_skaters_obj=skaters_df_2021)

NUM_OF_SEASONS_AGO_GOAL_DATA = 2

df_train = pd.concat([skaters_df_2018.get_df_for_goal_predictions(num_of_previous_seasons_for_goals_data=NUM_OF_SEASONS_AGO_GOAL_DATA),
                      skaters_df_2019.get_df_for_goal_predictions(num_of_previous_seasons_for_goals_data=NUM_OF_SEASONS_AGO_GOAL_DATA),
                      skaters_df_2020.get_df_for_goal_predictions(num_of_previous_seasons_for_goals_data=NUM_OF_SEASONS_AGO_GOAL_DATA),
                      skaters_df_2021.get_df_for_goal_predictions(num_of_previous_seasons_for_goals_data=NUM_OF_SEASONS_AGO_GOAL_DATA)], ignore_index=True)
df_eval = skaters_df_2022.get_df_for_goal_predictions(num_of_previous_seasons_for_goals_data=NUM_OF_SEASONS_AGO_GOAL_DATA)
y_train = df_train.pop('I_F_goals')
y_eval = df_eval.pop('I_F_goals')

print(df_eval)

CATEGORICAL_COLUMNS = ['position']
NUMERIC_COLUMNS = ['games_played', 'I_F_xGoals', '5on4_icetime', f'{NUM_OF_SEASONS_AGO_GOAL_DATA - 1}_season_ago_shooting_talent',
                   f'{NUM_OF_SEASONS_AGO_GOAL_DATA}_season_ago_shooting_talent', f'{NUM_OF_SEASONS_AGO_GOAL_DATA - 1}_season_ago_goals',
                   f'{NUM_OF_SEASONS_AGO_GOAL_DATA}_season_ago_goals']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df_train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


def make_input_function(data_df, label_df, num_epochs=250, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


train_input_fn = make_input_function(df_train, y_train)
eval_input_fn = make_input_function(df_eval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

linear_est.train(train_input_fn)

predictions = list(linear_est.predict(input_fn=eval_input_fn))
result = linear_est.evaluate(eval_input_fn)

clear_output()

print("Mean squared error:", result['average_loss'])

player_goal_predictions = []

pd.set_option('display.max_columns', 20)
specific_name = 'Zach Hyman'
row = df_eval.loc[df_eval['name'] == specific_name]
print(row)


for i, prediction in enumerate(predictions):
    player_data = df_eval.iloc[i]
    player_name = player_data['name']
    predicted_goals = prediction['predictions'][0]
    player_goal_predictions.append((player_name, predicted_goals))

sorted_player_goal_predictions = sorted(player_goal_predictions, key=lambda x: x[1], reverse=True)

i = 1
for player_name, predicted_goals in sorted_player_goal_predictions:
    print(f"{i}: {player_name}: {predicted_goals:.2f} goals")
    i += 1
