# TargetEncodingWithSmoothing
TargetEncodingとSmoothingを多分類対応で行います。

# 実装例
```
import pandas as pd 


training_df = pd.read_csv('./training_dataset.csv', sep=',')
prediction_df = pd.read_csv('./prediction_dataset.csv', sep=',')
smted = SmoothlyTargetEncodingDataloader(training_df,  prediction_df, '目的変数のカラム名')

 smted.get_training_dataset()  # 訓練データが手に入る
 smted.get_labels()  # 教師データが手に入る
```
