import pandas as pd
import numpy as np
import random

brands = [
    'Honda', 'Bajaj', 'Hero', 'Royal Enfield', 'Yamaha', 'Suzuki', 'TVS', 'KTM', 'Other'
]
cities = [
    'Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad', 'Ahmedabad', 'Jaipur', 'Lucknow'
]
owners = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner']

bike_names = [
    'Shine 125cc', 'Pulsar 150cc', 'Splendor Plus', 'Classic 350', 'FZ S', 'Access 125', 'Apache RTR 160', 'Duke 200', 'Other Model'
]

rows = 10000

data = []
for _ in range(rows):
    brand = random.choice(brands)
    city = random.choice(cities)
    owner = random.choice(owners)
    bike_name = f"{brand} {random.choice(bike_names)}"
    age = np.random.randint(1, 20)
    power = np.random.choice([100, 110, 125, 150, 160, 180, 200, 220, 350, 500, 650])
    kms_driven = np.random.randint(1000, 100000)
    price = int((power * 1000) - (age * 1000) - (kms_driven * 0.2) + np.random.randint(-5000, 5000))
    price = max(price, 5000)  # Minimum price
    data.append([
        bike_name, price, city, kms_driven, owner, age, power, brand
    ])

columns = ['bike_name', 'price', 'city', 'kms_driven', 'owner', 'age', 'power', 'brand']
df = pd.DataFrame(data, columns=columns)
df.to_csv('synthetic_bike_data.csv', index=False)
print('Generated synthetic_bike_data.csv with 10,000 rows.') 