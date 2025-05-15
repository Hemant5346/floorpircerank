# === app/model.py ===
import pandas as pd
import numpy as np

def preprocess_and_predict(new_data: pd.DataFrame, model, domain_means, country_freq, X_train_columns):
    new_data['Domain_te'] = new_data['Domain'].map(domain_means).fillna(domain_means.mean())
    new_data['Country_freq'] = new_data['Country'].map(country_freq).fillna(0)
    new_data_encoded = pd.get_dummies(new_data, columns=['Browser', 'Os'], drop_first=True)
    new_data_encoded = new_data_encoded.reindex(columns=X_train_columns, fill_value=0)
    if 'Country' in new_data_encoded:
        new_data_encoded.drop(columns=['Country'], inplace=True)
    if 'Domain' in new_data_encoded:
        new_data_encoded.drop(columns=['Domain'], inplace=True)
    pred = model.predict(new_data_encoded)
    return np.round(pred).astype(int)