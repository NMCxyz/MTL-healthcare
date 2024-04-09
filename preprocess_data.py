import pandas as pd

from utils import preprocess_data, resolution
from augment import *


train_path = "data/cls_regr_16_person/train.csv"
test_path = "data/cls_regr_16_person/test.csv"
len_point = 3000
resolution_param = 10
col_breath_rate = "breath_rate"

train_df = pd.read_csv(train_path, index_col=False)
test_df = pd.read_csv(test_path, index_col=False)

X_train, Y_train = preprocess_data(
    df=train_df,
    len_point=len_point,
    resolution_param=resolution_param,
    col_breath_rate=col_breath_rate
)
X_test, Y_test = preprocess_data(
    df=test_df,
    len_point=len_point,
    resolution_param=resolution_param,
    col_breath_rate=col_breath_rate
)

from augment import Augment

augmenter = Augment(features=X_train, labels=Y_train)
x_train, y_train = augmenter(augment_times=3)

X_train.shape

import numpy as np
data = {
    "x_train": x_train,
    "y_train": y_train,
    "x_test": X_test,
    "y_test": Y_test
}

np.savez("data/res_and_16pos.npz", **data)