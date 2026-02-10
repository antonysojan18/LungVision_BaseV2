import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP
MODEL_FILE = 'lung_cancer_model.pkl'
DATA_FILE = 'cancer patient datasets.csv'

print("📊 --- MODEL AUDIT STARTED ---")

# 2. LOAD DATA
if not os.path.exists(DATA_FILE):
    print("❌ Error: Dataset file not found.")
    exit()

df = pd.read_csv(DATA_FILE)
# ✅ Add errors='ignore' so it skips missing columns
X = df.drop(['Level', 'index', 'Patient Id'], axis=1, errors='ignore')
y = df['Level']

# Encode targets (Low/Medium/High -> 0/1/2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. SPLIT DATA (Must match app.py exactly!)
# We hide 20% of the data to use as a "Final Exam"
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. LOAD THE BRAIN
if os.path.exists(MODEL_FILE):
    print(f"🧠 Loading saved model: {MODEL_FILE}")
    with open(MODEL_FILE, 'rb') as f:
        saved_data = pickle.load(f)
        model = saved_data['model']
else:
    print("❌ Model not found. Run app.py first to train it!")
    exit()

# 5. THE EXAM (Prediction)
print("📝 Running predictions on Test Data...")
y_pred = model.predict(X_test)

# 6. RESULTS
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n🏆 MODEL ACCURACY: {acc:.2f}%")
print("-" * 30)
print("Detailed Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7. GENERATE VISUAL PROOF (Confusion Matrix)
# This shows exactly where the model got confused (if at all)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Acc: {acc:.2f}%)')
plt.show()