import pandas as pd
import numpy as np


class SmoothlyTargetEncodingDataloader:

    def __init__(self, training_df, prediction_df, target_column, k=0, f=1):
        training_df[target_column] = training_df[target_column].astype(np.str)
        self._training_df, self._smoothing_table, self._category_columns = \
            self._init_training_dataframe(training_df, target_column, k=k, f=f)
        self._labels = self._training_df[target_column]
        self._training_df = self._training_df.drop(target_column, axis=1).copy()
        self._prediction_df = \
            self._init_prediction_dataframe(
                prediction_df,
                self._smoothing_table,
                self._category_columns,
                self._training_df.columns)

    def get_training_dataset(self):
        return self._training_df

    def get_labels(self):
        return self._labels

    def get_prediction_dataset(self):
        return self._prediction_df

    def _init_training_dataframe(self, training_df, target_column, k=0, f=1):
        one_hot_targets, categories = self._get_target_and_categories(training_df, target_column)
        _template_df = pd.concat([one_hot_targets, categories], axis=1)
        ret_training_df = training_df.drop(categories.columns, axis=1)
        smoothing_table = categories.copy()

        for i, target_column_name in enumerate(one_hot_targets.columns):
            if i == len(one_hot_targets.columns) - 1:
                break
            one_hot_rate = np.sum(one_hot_targets[target_column_name]) / len(one_hot_targets[target_column_name])

            for category_column_name in categories.columns:
                categories_num_vectors = \
                    _template_df.groupby(category_column_name)[target_column_name].transform('size')
                category_mean = \
                    _template_df.groupby(category_column_name)[target_column_name].transform('mean').copy()
                smoothed_frame = self._calc_smoothing(categories_num_vectors, category_mean, one_hot_rate, k=k, f=f)
                ret_training_df['encoded_' + category_column_name + '_' + str(i)] = smoothed_frame
                smoothing_table['encoded_' + category_column_name + '_' + str(i)] = smoothed_frame
        return ret_training_df, smoothing_table, categories.columns

    def _init_prediction_dataframe(self, prediction_df, smoothing_table, category_columns, training_columns):
        ret_prediction_df = prediction_df.copy()
        for category_column_name in category_columns:
            ret_prediction_df = \
                pd.merge(
                    ret_prediction_df,
                    smoothing_table.loc[:, smoothing_table.columns.str.contains(category_column_name)].drop_duplicates()
                ).copy().drop(category_column_name, axis=1)
        ret_prediction_df = ret_prediction_df[training_columns]
        return ret_prediction_df

    def _get_target_and_categories(self, df, target_column):
        one_hot_targets = pd.get_dummies(df[target_column])
        df_without_target = df.drop(target_column, axis=1).copy()
        categories = df_without_target.select_dtypes(include=object).copy()
        return one_hot_targets, categories

    def _sigmoid(self, n, k=0, f=1):
        return 1 / (1 + np.exp(-((n - k) / f)))

    def _calc_smoothing(self, categories_num_vectors, category_mean, one_hot_rate, k=0, f=1):
        p = self._sigmoid(categories_num_vectors, k=k, f=f)
        return category_mean * p + one_hot_rate * (1 - p)
