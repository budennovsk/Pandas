# import tensorflow_decision_forests as tfdf
import pandas as pd

dataset = pd.read_csv("dataset.csv")
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")
print(tf_dataset)
# model = tfdf.keras.RandomForestModel()
# model.fit(tf_dataset)
#
# print(model.summary())