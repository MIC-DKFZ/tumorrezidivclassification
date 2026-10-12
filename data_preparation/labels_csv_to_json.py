import pandas as pd
import json
import os

# --- CONFIGURE THIS ---
# Path to your simple.csv
CSV_PATH = "/path/to/nnunet_preprocessed/Dataset0001_GlioblastomaTimeSeries/simple.csv"

# Path to your preprocessed data folder
PREPROCESSED_DIR = "/path/to/nnunet_preprocessed/Dataset0001_GlioblastomaTimeSeries"

# The prefix for your case identifiers
DATASET_PREFIX = "GlioblastomaTimeSeries"
# --- END CONFIGURATION ---

# Read the CSV
df = pd.read_csv(CSV_PATH)

# Create the dictionary in the required format
# Maps "GlioblastomaTimeSeries_001" -> 1, etc.
labels_dict = {
    f"{DATASET_PREFIX}_{int(row['PatID']):04d}": int(row['label'])
    for index, row in df.iterrows()
}

# Define the output path
output_path = os.path.join(PREPROCESSED_DIR, "labels.json")

# Save the dictionary as a JSON file
with open(output_path, 'w') as f:
    json.dump(labels_dict, f, indent=4)

print(f"Successfully created labels.json at: {output_path}")