import pickle
import json

import numpy as np

for i in range(5):
    
    # Load data from the pickle file
    with open(f'fordynesty/{i:03}_params.p', 'rb') as f:
        data = pickle.load(f)

    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = list(data[key])
        
    print('data', data)
        
    # Convert the data to JSON format and save it to a file
    with open(f'fordynesty/{i:03}_params.json', 'w') as f:
        json.dump(data, f, indent=4)
