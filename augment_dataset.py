import pandas as pd
import numpy as np

# ==========================================
# 1. SETTINGS
# ==========================================
TARGET_SIZE = 5000  # We want 5,000 patients total
INPUT_FILE = 'cancer patient datasets.csv' # The file you just renamed
OUTPUT_FILE = 'cancer_dataset_5000.csv'

# ==========================================
# 2. LOAD ORIGINAL DATA
# ==========================================
print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
current_size = len(df)
print(f"Current Size: {current_size} patients")

if current_size >= TARGET_SIZE:
    print("You already have enough data! No need to augment.")
    exit()

# ==========================================
# 3. GENERATE NEW PATIENTS (Smart Augmentation)
# ==========================================
print(f"Generating {TARGET_SIZE - current_size} new synthetic patients...")

new_rows = []
rows_to_generate = TARGET_SIZE - current_size

# We loop through existing patients and create "variations" of them
for i in range(rows_to_generate):
    # Pick a random real patient to base this new one on
    base_patient = df.sample(1).iloc[0].to_dict()
    
    # Create a new patient dict
    new_patient = base_patient.copy()
    
    # --- ADD "SMART NOISE" (Variation) ---
    
    # 1. Vary Age (± 2 years, but keep between 10 and 80)
    age_shift = np.random.randint(-2, 3)
    new_patient['Age'] = max(10, min(80, base_patient['Age'] + age_shift))
    
    # 2. Vary Severity Inputs (± 1 point, keep between 1-8)
    # This simulates different doctors giving slightly different ratings
    symptom_cols = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 
                    'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 
                    'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 
                    'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 
                    'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 
                    'Dry Cough', 'Snoring']
    
    for col in symptom_cols:
        if col in new_patient:
            shift = np.random.randint(-1, 2) # -1, 0, or +1
            new_val = base_patient[col] + shift
            new_patient[col] = max(1, min(8, new_val)) # Clamp between 1 and 8
            
    # 3. Recalculate "Years of Smoking" (Using our Medical Logic)
    # This ensures the new patient makes medical sense
    max_possible = max(0, new_patient['Age'] - 13)
    if new_patient['Level'] == 'High':
        years = np.random.randint(10, 26)
    elif new_patient['Level'] == 'Medium':
        years = np.random.randint(5, 16)
    else:
        years = np.random.randint(0, 6)
    new_patient['Years of Smoking'] = min(years, max_possible)
    
    # Add to list
    new_rows.append(new_patient)

# ==========================================
# 4. SAVE THE MEGA DATASET
# ==========================================
# Combine old + new
new_df = pd.DataFrame(new_rows)
final_df = pd.concat([df, new_df], ignore_index=True)

# Final Shuffle so old and new are mixed
final_df = final_df.sample(frac=1).reset_index(drop=True)

print(f"✅ Created {len(final_df)} unique patient records.")
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Saved to: {OUTPUT_FILE}")