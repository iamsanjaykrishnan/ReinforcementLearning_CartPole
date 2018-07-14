import numpy as np
import logging
import pandas as pd
import os

data = 'E:\ReinforcementLearning_CartPole\\NN_data\pdFrame.csv'
df = pd.DataFrame.from_csv(data)
OldState = df.as_matrix(['Old_State0', 'Old_State1', 'Old_State2', 'Old_State3'])
print(OldState.shape)

