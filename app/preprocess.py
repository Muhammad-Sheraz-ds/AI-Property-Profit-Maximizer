import numpy as np

def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocess input data for prediction.

    Args:
        data (dict): Input data from the user as a dictionary.

    Returns:
        np.ndarray: Preprocessed data as a NumPy array.
    """
    # Arrange the data in the exact order and format required by the pipeline
    input_array = np.array([
        float(data['crime_rate']),
        data['renovation_level'],
        int(data['num_rooms']),
        int(data['Property']),
        data['amenities_rating'],
        float(data['carpet_area']),
        float(data['property_tax_rate']),
        data['Locality'],
        data['Residential'],
        float(data['Estimated Value'])
    ], dtype=object).reshape(1, -1)

    return input_array
