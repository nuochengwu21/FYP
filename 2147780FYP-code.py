import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from xgboost import XGBClassifier

# 读取数据
neg_data = pd.read_csv('neg.csv')
pos_data = pd.read_csv('pos.csv')

# 构建X, y
X_pos = pos_data.iloc[:, 6:]
y_pos = np.ones(X_pos.shape[0])
X_neg = neg_data.iloc[:, 6:]
y_neg = np.zeros(X_neg.shape[0])
X = pd.concat([X_pos, X_neg], axis=0)
y = np.concatenate([y_pos, y_neg])

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled['Group'] = ['m7G'] * len(X_pos) + ['NormalG'] * len(X_neg)

# 3.3部分：验证信号和错误的差异性
custom_names = [
    "Current_Mean_Center", "Current_SD_Center", "Dwell_Time_Center",
    "Current_Mean_-2", "Current_Mean_-1", "Current_Mean_+1",
    "Current_SD_+1", "Dwell_Time_-1", "Dwell_Time_+1", "Dwell_Time_-2",
    "Mismatch_Count", "Deletion_Count", "Insertion_Count", 
    "Low_Quality_Base_Count", "Other_Error_Feature"
]
feature_cols = X.columns[:15]
results = []

for i, col in enumerate(feature_cols):
    m7g_vals = df_scaled[df_scaled['Group'] == 'm7G'][col]
    g_vals = df_scaled[df_scaled['Group'] == 'NormalG'][col]
    stat, p = mannwhitneyu(m7g_vals, g_vals, alternative='two-sided')
    results.append({'Feature': custom_names[i], 'p-value': p})

    # 绘图
    plt.figure()
    sns.violinplot(x='Group', y=col, data=df_scaled, palette='Set2', cut=0)
    plt.title(f'{custom_names[i]} - p={p:.4e}')
    plt.tight_layout()
    plt.savefig(f'{custom_names[i]}.png')
    plt.close()

# 保存统计表格
pd.DataFrame(results).to_csv('feature_group_comparison.csv', index=False)

# 分割 & SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 训练XGBoost模型
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train_bal, y_train_bal)

# SHAP分析
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_test)
shap.summary_plot(shap_vals, X_test, feature_names=X.columns)

# 多模型评估
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}

metrics_dict = {}
for name, clf in models.items():
    clf.fit(X_train_bal[:len(y_train)//20], y_train_bal[:len(y_train)//20])
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics_dict[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba),
        'F1': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Avg Precision': average_precision_score(y_test, y_proba)
    }

pd.DataFrame(metrics_dict).T.to_csv('model_performance.csv')

# 可视化：ROC曲线
plt.figure(figsize=(8, 6))
for name, clf in models.items():
    y_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# 可视化：PR曲线
plt.figure(figsize=(8, 6))
for name, clf in models.items():
    y_proba = clf.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(rec, prec)
    plt.plot(rec, prec, label=f'{name} (AUC={pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig('pr_curve.png')
plt.close()
