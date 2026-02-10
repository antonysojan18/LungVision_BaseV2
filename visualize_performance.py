import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# 1. LOAD DATA (Safely)
# ==========================================
print("Loading dataset...")
df = pd.read_csv('cancer patient datasets.csv')

# Convert Target to Numbers
target_map = {'Low': 0, 'Medium': 1, 'High': 2}
if 'Level' in df.columns:
    df['Level_Num'] = df['Level'].map(target_map)

# Drop non-features safely (The Fix you know!)
X = df.drop(columns=['Level', 'Level_Num', 'index', 'Patient Id'], errors='ignore')
y = df['Level_Num']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. DEFINE THE CONTENDERS
# ==========================================
print("Training models individually to compare...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Add the Ensemble (The Team)
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft'
)
models["Ensemble (Final Model)"] = ensemble

# ==========================================
# 3. RUN THE COMPETITION
# ==========================================
results = []

for name, model in models.items():
    print(f"-> Testing {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    results.append({'Model': name, 'Accuracy': acc})

# ==========================================
# 4. GENERATE THE GRAPH
# ==========================================
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=True)

# Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Create Bar Plot
chart = sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')

# Add Labels on the bars
for i, p in enumerate(chart.patches):
    width = p.get_width()
    plt.text(width + 0.5,       # x-position
             p.get_y() + p.get_height()/2 + 0.1, # y-position
             f'{width:.2f}%',   # Label text
             ha="left")

plt.title('Performance Comparison: Individual Models vs. Ensemble', fontsize=15)
plt.xlabel('Accuracy (%)', fontsize=12)
plt.xlim(80, 105) # Zoom in to show differences clearly
plt.tight_layout()

# Save and Show
plt.savefig('model_comparison_graph.png')
print("\n✅ Graph saved as 'model_comparison_graph.png'")
plt.show()