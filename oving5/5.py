import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# |%%--%%| <6Q5BwBGWTC|lUWbVjNie7>
r"""°°°
# Exercise 5
1. Download the mushroom dataset here: https://archive.ics.uci.edu/ml/datasets/Mushroom
2. Create a new Jupyter notebook
3. Load the dataset from CSV into pandas
4. Explore the distributions in the data. For example, how is habitat distributed between edibility vs non-edibility?
5. The data is entirely categorical. Convert each feature to dummy variables.
6. Visualise the feature space using a similar method to the one we used for the 20 newgroups dataset.

Your submission should be your Jupyter notebook, keep it short and concise.
°°°"""
# |%%--%%| <lUWbVjNie7|daVb7E5FBo>
r"""°°°
# Task 1 and Task 2
1. Download the mushroom dataset here: https://archive.ics.uci.edu/ml/datasets/Mushroom
2. Create a new Jupyter notebook

Yeah I did it
°°°"""
# |%%--%%| <daVb7E5FBo|G9E46aKAqK>
r"""°°°
# Task 3
3. Load the dataset from CSV into pandas
°°°"""
# |%%--%%| <G9E46aKAqK|LoOG93WQnZ>

mushroom_repo_id = 73
url = f'https://archive.ics.uci.edu/static/public/{mushroom_repo_id}/data.csv'
df = pd.read_csv(url)
df
# |%%--%%| <LoOG93WQnZ|O4Xphadwv1>
r"""°°°

# Task 4
4. Explore the distributions in the data. For example, how is habitat
distributed between edibility vs non-edibility?

NOTE: Probably need to plot edibility with habitat
°°°"""
# |%%--%%| <O4Xphadwv1|6hVm8Dtoq9>
x = df.groupby(['poisonous', 'habitat']).size()
x
# |%%--%%| <6hVm8Dtoq9|Zhqr6DCOzL>
r"""°°°
# Task 5
5. The data is entirely categorical. Convert each feature to dummy variables.
°°°"""

# |%%--%%| <Zhqr6DCOzL|ZPGXPUgorf>
df_encoded = pd.get_dummies(df)
df_encoded
# |%%--%%| <ZPGXPUgorf|PwZqg8MCBO>
r"""°°°
# Task 6
6. Visualise the feature space using a similar method to the one we used for the 20 newgroups dataset.
°°°"""
# |%%--%%| <PwZqg8MCBO|pVN9LNabDA>
plt.spy(df_encoded, markersize=1)
fig = plt.gcf()
fig.set_size_inches(60, 250)
plt.plot()
plt.show()

# |%%--%%| <pVN9LNabDA|mdFdZks2N6>
