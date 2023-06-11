def load_dataset(latitude, longitude, start_date, end_date, code):
    if code == 0:
        dataset = 'reference_data'
    else:
        dataset = 'prediction_training_data'
    url = 'https://archive-api.open-meteo.com/v1/archive'

