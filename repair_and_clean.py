import pandas as pd
import numpy as np
import joblib
import os

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
TARGET_ROWS = 4781
MIN_START_AGE = 16         # ✅ NEW RULE: Must be at least 16 to start smoking
INPUT_FILE = 'cancer patient datasets.csv'
OUTPUT_CSV = 'cancer patient datasets.csv'
MODEL_FILE = 'lungvision_ensemble_model.pkl'

# ==========================================
# 1. BULLETPROOF LOADING
# ==========================================
print(f"\n🔄 [1/5] Loading {INPUT_FILE}...")

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found. Please restore your file.")
    exit()

df = pd.read_csv(INPUT_FILE)

# ==========================================
# 1.5 CLEANING (Removing Junk)
# ==========================================
print("   -> [Cleaning] Removing irrelevant data columns...")

# ✅ We KEEP 'Gender' (Essential). We only drop the junk IDs.
irrelevant_cols = [
    'index', 
    'Patient Id', 
    'Patient ID', 
    'id', 
    'Unnamed: 0'
]

# Safe Drop
cols_to_drop = [c for c in irrelevant_cols if c in df.columns]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"   -> Dropped junk columns: {cols_to_drop}")

# Drop Duplicates
df = df.drop_duplicates(keep='first')

# ==========================================
# 2. MEDICAL LOGIC REPAIR (The Strict Rule)
# ==========================================
print(f"   -> [2/5] Enforcing Strict Logic (Min Smoking Age: {MIN_START_AGE})...")

def repair_patient(row):
    # 1. Clamp Symptoms (1-8)
    skip_cols = ['Age', 'Gender', 'Level', 'Years of Smoking', 'Level_Num']
    for col in row.index:
        if col not in skip_cols and isinstance(row[col], (int, float)):
             row[col] = max(1, min(8, row[col]))

    # 2. Fix Smoking Years (The New Rule)
    # Rule: Age - Years of Smoking must be >= 16
    # Therefore: Years of Smoking <= Age - 16
    max_possible = max(0, int(row['Age']) - MIN_START_AGE)
    
    current = row.get('Years of Smoking', 0)
    
    # Logic: If data violates the rule (started too young), we clamp it down.
    if pd.isna(current) or current > max_possible:
        # If we need to generate a new number, we respect the limit
        if row['Level'] == 'High':
            generated = np.random.randint(10, 25)
        elif row['Level'] == 'Medium':
            generated = np.random.randint(5, 15)
        else:
            generated = np.random.randint(0, 5)
        
        # STRICT CLAMP: Even the generated number cannot break the age rule
        row['Years of Smoking'] = min(generated, max_possible)
    else:
        # Even if the number exists, clamp it if it violates the rule
        row['Years of Smoking'] = min(current, max_possible)
        
    return row

df = df.apply(repair_patient, axis=1)

# ==========================================
# 3. AUGMENTATION
# ==========================================
print(f"   -> [3/5] Augmenting to {TARGET_ROWS} rows...")
needed = TARGET_ROWS - len(df)

if needed > 0:
    new_patients = []
    for _ in range(needed):
        parent = df.sample(1).iloc[0].copy()
        
        # Mutate Age
        parent['Age'] = max(18, min(80, parent['Age'] + np.random.randint(-3, 4)))
        
        # Mutate Symptoms
        for col in parent.index:
            if col not in ['Age', 'Gender', 'Level', 'Years of Smoking', 'Level_Num']:
                val = parent[col]
                if isinstance(val, (int, float)):
                    parent[col] = max(1, min(8, val + np.random.choice([-1, 0, 1])))
        
        # Repair the new patient to ensure they follow the 16-year rule
        parent = repair_patient(parent)
        new_patients.append(parent)
    
    df_new = pd.DataFrame(new_patients)
    df = pd.concat([df, df_new], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)
print(f"      ✅ Saved strict data to '{OUTPUT_CSV}'")

# ==========================================
# 4. TRAIN MODEL
# ==========================================
print("\n🧠 [4/5] Training The Council of AIs...")

target_map = {'Low': 0, 'Medium': 1, 'High': 2}
if 'Level' in df.columns:
    df['Level_Num'] = df['Level'].map(target_map)

# Drop non-features safely
X = df.drop(columns=['Level', 'Level_Num'], errors='ignore')
y = df['Level_Num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

ensemble = VotingClassifier(estimators=estimators, voting='soft')
ensemble.fit(X_train, y_train)

# ==========================================
# 5. REPORT
# ==========================================
acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"\n🚀 FINAL MODEL ACCURACY: {acc*100:.2f}%")
joblib.dump(ensemble, MODEL_FILE)
print("✅ System Ready.")