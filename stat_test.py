import pandas as pd
from scipy import stats
import re

file_name = "all_limited_features.csv"
df = pd.read_csv(file_name)

df_filtered = df[df['season'] == 'summer1'].copy()

feature_columns = [col for col in df_filtered.columns if re.match(r'B\d+_.+', col)]
# 

anova_results = []
cultures_to_test = df_filtered['culture'].unique()

for feature in feature_columns:
    groups = [df_filtered[df_filtered['culture'] == c][feature].dropna() for c in cultures_to_test]

    # ANOVA se tiver membro suficiente
    if all(len(g) > 1 for g in groups):
        f_statistic, p_value = stats.f_oneway(*groups)
        anova_results.append({
            'feature': feature,
            'F_Statistic': f_statistic,
            'P_Value': p_value
        })

results_df = pd.DataFrame(anova_results)
results_df_sorted = results_df.sort_values(by='F_Statistic', ascending=False).reset_index(drop=True)

results_df_sorted.to_csv("anova_distance_features.csv", index=False)