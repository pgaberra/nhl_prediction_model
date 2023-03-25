import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from sklearn.preprocessing import StandardScaler
from skaters_stats_df import SkatersStats


skaters_df_2018 = SkatersStats('data/skaters_2018.csv', '2018')
skaters_df_2019 = SkatersStats('data/skaters_2019.csv', '2019')
skaters_df_2020 = SkatersStats('data/skaters_2020.csv', '2020')
skaters_df_2021 = SkatersStats('data/skaters_2021.csv', '2021')
skaters_df_2022 = SkatersStats('data/skaters_2022.csv', '2022')


scaler = StandardScaler()

df_train = pd.concat(
    [skaters_df_2018.get_df_for_assist_predictions(),
     skaters_df_2019.get_df_for_assist_predictions(),
     skaters_df_2020.get_df_for_assist_predictions(),
     skaters_df_2021.get_df_for_assist_predictions()],
    ignore_index=True)
df_eval = skaters_df_2022.get_df_for_assist_predictions()
y_train = df_train.pop('I_F_assists')
y_eval = df_eval.pop('I_F_assists')

CATEGORICAL_COLUMNS = ['playerId', 'position']
NUMERIC_COLUMNS = ['icetime', 'games_played', 'OnIce_F_xGoals', 'OnIce_F_shotsOnGoal', '5on4_icetime', 'onIce_corsiPercentage']

# Fit the scaler on the training data
scaler.fit(df_train[NUMERIC_COLUMNS])

# Transform the training and evaluation data using the scaler
scaled_train_data = scaler.transform(df_train[NUMERIC_COLUMNS])
scaled_eval_data = scaler.transform(df_eval[NUMERIC_COLUMNS])

df_train[NUMERIC_COLUMNS] = scaled_train_data
df_eval[NUMERIC_COLUMNS] = scaled_eval_data

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df_train[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_function(data_df, label_df, num_epochs=350, shuffle=True, batch_size=32):
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

print(result)

player_assists_predictions = []

for i, prediction in enumerate(predictions):
    player_data = df_eval.iloc[i]
    player_name = player_data['name']
    predicted_assists = prediction['predictions'][0]

    # Set negative predictions to 0
    if predicted_assists < 0:
        predicted_assists = 0

    player_assists_predictions.append((player_name, predicted_assists))

sorted_player_assists_predictions = sorted(player_assists_predictions, key=lambda x: x[1], reverse=True)

i = 1
for player_name, predicted_assists in sorted_player_assists_predictions:
    print(f"{i}: {player_name}: {predicted_assists:.2f} assists")
    i += 1
