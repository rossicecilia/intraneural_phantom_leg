import numpy as np

def extracting_array_from_excel(df, row):
    results = np.array([el for el in df.iloc[row] if el != np.NaN and el not in ['Ankle', 'Knee', 'Toes','Flexion', 'Extension']])
    # Remove NaN values
    return results[~np.isnan(results)]