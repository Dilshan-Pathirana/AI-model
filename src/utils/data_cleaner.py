import pandas as pd

def clean_data(data):
    """
    Clean and preprocess the fetched data.
    """
    # Drop any rows with missing values
    data = data.dropna()

    # Drop duplicates
    data = data.drop_duplicates()

    # ✅ Ensure 'date' column exists and is in datetime format
    if 'date' not in data.columns:
        data['date'] = data.index

    data['date'] = pd.to_datetime(data['date'])

    return data
