import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler
from skaters_stats_df import SkatersStats


def predict(get_df_function, target_column, categorical_columns, numeric_columns, num_epochs):
    skaters_df_2019 = SkatersStats('data/skaters_2019.csv', '2019')
    skaters_df_2020 = SkatersStats('data/skaters_2020.csv', '2020')
    skaters_df_2021 = SkatersStats('data/skaters_2021.csv', '2021')
    skaters_df_2022 = SkatersStats('data/skaters_2022.csv', '2022')

    scaler = StandardScaler()

    df_train = pd.concat(
        [getattr(skaters_df_2019, get_df_function)(),
         getattr(skaters_df_2020, get_df_function)(),
         getattr(skaters_df_2021, get_df_function)(),
         getattr(skaters_df_2022, get_df_function)()],
        ignore_index=True)
    df_eval = getattr(skaters_df_2022, get_df_function)()

    y_train = df_train.pop(target_column)
    y_eval = df_eval.pop(target_column)

    # Fit the scaler on the training data
    scaler.fit(df_train[numeric_columns])

    # Transform the training and evaluation data using the scaler
    scaled_train_data = scaler.transform(df_train[numeric_columns])
    scaled_eval_data = scaler.transform(df_eval[numeric_columns])

    df_train[numeric_columns] = scaled_train_data
    df_eval[numeric_columns] = scaled_eval_data

    feature_columns = []

    for feature_name in categorical_columns:
        vocabulary = df_train[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in numeric_columns:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    train_input_fn = make_input_function(df_train, y_train, num_epochs=num_epochs)
    eval_input_fn = make_input_function(df_eval, y_eval, num_epochs=1, shuffle=False)

    linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    linear_est.train(train_input_fn)

    predictions = list(linear_est.predict(input_fn=eval_input_fn))

    clear_output()

    players = []

    for i, prediction in enumerate(predictions):
        player = {}
        player_data = df_eval.iloc[i]
        player_id = player_data['playerId']
        player_name = player_data['name']
        player_age = player_data['age']
        predicted_target = prediction['predictions'][0]

        # Set negative predictions to 0
        if predicted_target < 0:
            predicted_target = 0

        player['playerId'] = player_id
        player['name'] = player_name
        player['age'] = player_age
        player['prediction'] = predicted_target
        players.append(player)

    return players


def make_input_function(data_df, label_df, num_epochs=350, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


def get_points_predictions():
    goal_predictions = predict('get_df_for_goal_predictions', 'I_F_goals', ['playerId', 'position', 'age'],
                               ['icetime', 'games_played', 'I_F_flurryAdjustedxGoals', '5on4_icetime', 'I_F_shotsOnGoal'], 200)
    assist_predictions = predict('get_df_for_assist_predictions', 'I_F_assists', ['playerId', 'position', 'age'],
                                 ['icetime', 'games_played', 'OnIce_F_flurryAdjustedxGoals', 'OnIce_F_shotsOnGoal', '5on4_icetime',
                                  'onIce_corsiPercentage'], 350)

    combined_predictions = []

    for goal_pred, assist_pred in zip(goal_predictions, assist_predictions):
        player_prediction = {
            'playerId': goal_pred['playerId'],
            'name': goal_pred['name'],
            'age': goal_pred['age'],
            'goal_prediction': round(goal_pred['prediction']),
            'assist_prediction': round(assist_pred['prediction']),
            'point_prediction': round(goal_pred['prediction']) + round(assist_pred['prediction'])
        }
        combined_predictions.append(player_prediction)

    return sorted(combined_predictions, key=lambda x: (x['point_prediction'], x['goal_prediction']), reverse=True)


if __name__ == '__main__':
    points_predictions = get_points_predictions()
    for i, player in enumerate(points_predictions, start=1):
        predicted_goals = player['goal_prediction']
        predicted_assists = player['assist_prediction']
        predicted_points = player['point_prediction']
        print(f'{i:3}: {player["name"]:20} {predicted_points:5} points ({predicted_goals}G {predicted_assists}A) {player["age"]}')
