import pandas as pd
import sys
import numpy as np

models_folders = [f"models{i}" for i in range(1, 6)]

# Criar os test_culture.csv
print(models_folders)

final = pd.DataFrame(columns=["feat1", "imp1",
                           "feat2", "imp2",
                           "feat3", "imp3",
                           "feat4", "imp4",
                           "feat5", "imp5"])

culture = "bean"

for i, model in enumerate(models_folders):
    df = pd.read_csv(f"{model}/features/feature_culture_correlation.csv")
    df.rename({"Unnamed: 0": "feature"}, axis=1, inplace=True)
    df = df[["feature", f'Label_{culture}']]
    df["abs"] = df[f'Label_{culture}'].abs()
    df = df.sort_values(by="abs", ascending=False)
    
    final[[f"feat{i+1}", f"imp{i+1}"]] = df[["feature", f"Label_{culture}"]]

    del df

print(final)
final.to_csv(f"test_{culture}.csv")

# Loadar os test_culture.csv

'''
features_to_analyze = []

cultures = ["bean", "maize", "rice", "soybean"]

features_dict = {
    "bean": [], "maize": [], "rice": [], "soybean": []
}

for culture in cultures:
    df = pd.read_csv(f"test_{culture}.csv", index_col=0)

    features_dict[f"{culture}"] = list(df["feat1"].head(15).values)

for val in features_dict.values():
    features_to_analyze.extend(val)

# print(features_to_analyze)
print(len(features_to_analyze))
print(features_to_analyze)

# sys.exit(1)

corr_matrix = pd.read_csv(f"models3/features/feature_feature_correlation_matrix.csv", index_col=0)

print(list(corr_matrix.columns))

for item in features_to_analyze:
    if item not in list(corr_matrix.columns):
        print(f"{item} not in it")

sys.exit(1)
filtered_corr = corr_matrix.loc[features_to_analyze, features_to_analyze]

import seaborn as sns
import matplotlib.pyplot as plt

# plt.figure(figsize=(16, 15))
# sns.heatmap(filtered_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True)
# plt.title('Filtered Correlation Matrix')
# plt.show()

mask = np.triu(np.ones(filtered_corr.shape), k=1).astype(bool)
stacked_corr = filtered_corr.where(mask).stack()
low_corr_features = stacked_corr[stacked_corr.abs() > 0.8].index.tolist()

feature_importances = pd.read_csv("models3/features/consolidated_feature_importances.csv")

print(feature_importances)


sett = list()

for item in low_corr_features:
    item1, item2 = item
    sett.append(item1)
    sett.append(item2)

to_drop = set()

for pair in low_corr_features:
    feat1, feat2 = pair
    print(feature_importances[feature_importances['Feature']==feat2]["Mean_Importance"].values)
    v1 = feature_importances[feature_importances['Feature']==feat1]["Mean_Importance"].values[0]
    print(feat2)
    v2 = feature_importances[feature_importances['Feature']==feat2]["Mean_Importance"].values[0]
    if v1>v2:
        to_drop.add(feat1)
    else:
        to_drop.add(feat2)

print(f"Have to drop {len(to_drop)} out of {len(features_to_analyze)}")

new_feats = [feat for feat in features_to_analyze if feat not in to_drop]

print(f"Size of new_feats: {len(new_feats)}")

new_filtered = corr_matrix.loc[new_feats, new_feats]

plt.figure(figsize=(16, 15))
sns.heatmap(new_filtered, annot=False, cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title('Filtered Correlation Matrix')
plt.show()

'''