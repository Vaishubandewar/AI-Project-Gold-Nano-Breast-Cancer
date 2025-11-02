# demo_random_forest.py
# Breast Cancer Detection using Gold Nanoparticles and AI
# Author: Vaishnavi Bandewar

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Dummy dataset
data = {
    'Size': [10, 15, 20, 25, 30],
    'Dose': [1.0, 1.2, 1.5, 1.8, 2.0],
    'Cell_Death%': [55, 65, 75, 85, 95]
}

df = pd.DataFrame(data)

# Train model
X = df[['Size', 'Dose']]
y = df['Cell_Death%']

model = RandomForestClassifier()
model.fit(X, y)

print("✅ Model trained successfully on dummy data.")
print("📊 Predicted Cell Death for new sample [Size=22, Dose=1.6]:")
print(model.predict([[22, 1.6]]))

