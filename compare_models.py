import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier  # <--- THE NEW CHALLENGER

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading dataset...")
df = pd.read_csv('cancer patient datasets.csv')

# Prepare Data
target_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Level'] = df['Level'].map(target_map)
X = df.drop(columns=['Level', 'Patient Id', 'index'], errors='ignore')
y = df['Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. DEFINE THE 6 CHALLENGERS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0) # verbose=0 keeps it quiet
}

# ==========================================
# 3. THE BATTLE
# ==========================================
results = []
print("\n--- MODEL BATTLE RESULTS ---")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    results.append({'Model': name, 'Accuracy': acc})
    print(f"-> {name}: {acc:.2f}%")

# ==========================================
# 4. VISUALIZE
# ==========================================
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Accuracy', y='Model', data=results_df, palette='magma') # Changed color to look cooler
plt.title('Model Accuracy Comparison (Includes CatBoost)')
plt.xlabel('Accuracy (%)')
plt.xlim(85, 100) 
plt.show()

print("\n🏆 THE WINNER IS:", results_df.iloc[0]['Model'])