# =====================================================================
# Data Science Project for Predictive Analysis
# Dataset: Data_Feb_2026 (Banknote Authentication)
# Tasks: Preprocessing & EDA | Classification (SVM) | Regression
# =====================================================================
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, r2_score,
                             mean_squared_error, mean_absolute_error)

sns.set_style("whitegrid")
RND = 42

# ------------------------- ROBUST DATA LOADER ------------------------
# Looks for the data file in the SAME folder as this script and
# accepts either .xlsx or .csv, so the path can never be "not found".
HERE = os.path.dirname(os.path.abspath(__file__))

def load_data():
    candidates = (
        glob.glob(os.path.join(HERE, "Data_Feb_2026.xlsx")) +
        glob.glob(os.path.join(HERE, "Data_Feb_2026.csv")) +
        glob.glob(os.path.join(HERE, "Data_Feb*2026*.xlsx")) +
        glob.glob(os.path.join(HERE, "Data_Feb*2026*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"Put Data_Feb_2026.xlsx (or .csv) next to this script.\n"
            f"Looked in: {HERE}\nFiles here: {os.listdir(HERE)}")
    path = candidates[0]
    print("Loading:", os.path.basename(path))
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    xl = pd.ExcelFile(path)                       # pick sheet 'in' if present
    sheet = "in" if "in" in xl.sheet_names else xl.sheet_names[0]
    return pd.read_excel(path, sheet_name=sheet)

# ------------------------- TASK 1: LOAD & CLEAN ----------------------
df = load_data()
print("Raw shape:", df.shape)

df = df.drop_duplicates()                          # 25 duplicate rows
df = df.dropna(subset=['Label'])                   # 1 missing target
df = df[~((df['X1'] == -70) | (df['X2'] == 120) |  # 1 corrupted row
          (df['X3'] == -50) | (df['X4'] == -80))].copy()
df['Label'] = df['Label'].astype(int)
print("Clean shape:", df.shape)
print("Class counts:", df['Label'].value_counts().to_dict())

feat = ['X1', 'X2', 'X3', 'X4']
names = {'X1': 'Variance', 'X2': 'Skewness', 'X3': 'Curtosis', 'X4': 'Entropy'}

# ------------------------- TASK 1: EDA -------------------------------
print(df.describe().round(3))
print(df.corr().round(3))

# Fig 1 - class distribution
ax = sns.countplot(x='Label', data=df, palette=['#2a9d8f', '#e76f51'])
ax.set_xticklabels(['Genuine (0)', 'Forged (1)']); ax.set_title('Class Distribution')
plt.tight_layout(); plt.show()

# Fig 2 - feature distributions by class
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, c in zip(axes.ravel(), feat):
    sns.histplot(data=df, x=c, hue='Label', bins=35, kde=True, ax=ax,
                 palette=['#2a9d8f', '#e76f51'], alpha=0.55)
    ax.set_title(f'{c} - {names[c]}')
plt.tight_layout(); plt.show()

# Fig 3 - correlation heatmap
plt.figure(figsize=(6.5, 5))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Correlation Heatmap'); plt.tight_layout(); plt.show()

# Fig 4 - boxplots by class
fig, axes = plt.subplots(1, 4, figsize=(13, 4))
for ax, c in zip(axes, feat):
    sns.boxplot(x='Label', y=c, data=df, ax=ax, palette=['#2a9d8f', '#e76f51'])
    ax.set_title(f'{c} - {names[c]}')
plt.tight_layout(); plt.show()

# Fig 5 - pairplot
sns.pairplot(df, vars=feat, hue='Label', palette=['#2a9d8f', '#e76f51'],
             diag_kind='kde', plot_kws={'alpha': 0.5, 's': 18})
plt.show()

# ------------------------- TASK 2a: CLASSIFICATION (SVM) -------------
X, y = df[feat], df['Label']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=RND, stratify=y)
scaler = StandardScaler().fit(Xtr)                 # scaling is vital for SVM
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=RND)
svm.fit(Xtr_s, ytr)
yp = svm.predict(Xte_s)
yprob = svm.predict_proba(Xte_s)[:, 1]

print("\n--- SVM CLASSIFICATION ---")
print("Accuracy :", round(accuracy_score(yte, yp), 4))
print("Precision:", round(precision_score(yte, yp), 4))
print("Recall   :", round(recall_score(yte, yp), 4))
print("F1-score :", round(f1_score(yte, yp), 4))
print("ROC-AUC  :", round(roc_auc_score(yte, yprob), 4))
print(confusion_matrix(yte, yp))
print(classification_report(yte, yp, target_names=['Genuine', 'Forged']))

# Fig 6 - confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(yte, yp), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Genuine', 'Forged'], yticklabels=['Genuine', 'Forged'])
plt.title('SVM Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout(); plt.show()

# Fig 7 - ROC curve
fpr, tpr, _ = roc_curve(yte, yprob)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, color='#e76f51', lw=2,
         label=f'SVM (AUC={roc_auc_score(yte, yprob):.3f})')
plt.plot([0, 1], [0, 1], '--', color='grey')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM'); plt.legend(loc='lower right')
plt.tight_layout(); plt.show()

# ------------------------- TASK 2b: REGRESSION ----------------------
# Predict skewness (X2) from variance, curtosis and entropy
Xr, yr = df[['X1', 'X3', 'X4']], df['X2']
Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.25, random_state=RND)

lin = LinearRegression().fit(Xrtr, yrtr)
ypl = lin.predict(Xrte)
print("\n--- LINEAR REGRESSION (target = X2 skewness) ---")
print("R2  :", round(r2_score(yrte, ypl), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(yrte, ypl)), 4))
print("MAE :", round(mean_absolute_error(yrte, ypl), 4))
print("Coefficients:", dict(zip(['X1', 'X3', 'X4'], lin.coef_.round(3))))

rf = RandomForestRegressor(n_estimators=300, random_state=RND).fit(Xrtr, yrtr)
ypr = rf.predict(Xrte)
print("\n--- RANDOM FOREST REGRESSION (comparison) ---")
print("R2  :", round(r2_score(yrte, ypr), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(yrte, ypr)), 4))

# Fig 8 - predicted vs actual
plt.figure(figsize=(5.5, 4.5))
plt.scatter(yrte, ypl, alpha=0.5, color='#264653', s=22)
lims = [yrte.min(), yrte.max()]
plt.plot(lims, lims, '--', color='#e76f51', lw=2)
plt.xlabel('Actual Skewness (X2)'); plt.ylabel('Predicted Skewness (X2)')
plt.title('Linear Regression - Predicted vs Actual')
plt.tight_layout(); plt.show()

# Fig 9 - residuals
plt.figure(figsize=(5.5, 4.5))
plt.scatter(ypl, yrte - ypl, alpha=0.5, color='#2a9d8f', s=22)
plt.axhline(0, color='#e76f51', ls='--', lw=2)
plt.xlabel('Predicted Skewness (X2)'); plt.ylabel('Residual')
plt.title('Residual Plot - Linear Regression')
plt.tight_layout(); plt.show()

# Fig 10 - regression model comparison
plt.figure(figsize=(6, 4))
r2s = [r2_score(yrte, ypl), r2_score(yrte, ypr)]
b = plt.bar(['Linear Reg.', 'Random Forest'], r2s, color=['#264653', '#2a9d8f'])
for bar, v in zip(b, r2s):
    plt.text(bar.get_x() + bar.get_width()/2, v, f'{v:.3f}',
             ha='center', va='bottom', fontweight='bold')
plt.ylabel('Test R2'); plt.ylim(0, 1); plt.title('Regression Model Comparison (R2)')
plt.tight_layout(); plt.show()

print("\nDone - all figures displayed.")