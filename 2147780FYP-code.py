import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, matthews_corrcoef, average_precision_score, precision_recall_curve, auc
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Load data
neg_data = pd.read_csv('neg.csv')
pos_data = pd.read_csv('pos.csv')

print("Positive Data Shape:", pos_data.shape)
print("Negative Data Shape:", neg_data.shape)

# Feature extraction
X_pos = pos_data.iloc[:, 6:]
y_pos = np.ones(X_pos.shape[0])
X_neg = neg_data.iloc[:, 6:]
y_neg = np.zeros(X_neg.shape[0])

X = pd.concat([X_pos, X_neg], axis=0)
y = np.concatenate([y_pos, y_neg])

print("Combined Feature Shape:", X.shape)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE, Training Set Shape:", X_train_balanced.shape)
print(f"Number of samples after SMOTE: {X_train_balanced.shape[0]}")

# Train XGBoost model
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train_scaled, y_train)

# SHAP explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# === 构建 posX_base_fY 命名方式 ===
features_per_base = X.shape[1] // 5
position = int(pos_data.iloc[0, 4])  # 第4列为中心位置
kmer = str(pos_data.iloc[0, 5])      # 第5列为5-mer序列

custom_feature_names = []
for i, base in enumerate(kmer):
    base_pos = position - 2 + i
    for j in range(features_per_base):
        custom_feature_names.append(f"pos{base_pos}_{base}_f{j}")

# === SHAP summary 图 ===
shap.summary_plot(shap_values, X_test_scaled, feature_names=custom_feature_names, show=True)

# === SHAP bar 图 ===
shap_abs = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': custom_feature_names,
    'SHAP Importance': shap_abs
}).sort_values(by='SHAP Importance', ascending=False)

plt.figure(figsize=(12, 7))
plt.barh(shap_df['Feature'], shap_df['SHAP Importance'], color='skyblue')
plt.xlabel('Mean Absolute SHAP Value (Feature Importance)')
plt.title('Feature Importance Based on SHAP Values')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Random sampling 1/20
sample_fraction = 1 / 20
sample_size = int(X_train_balanced.shape[0] * sample_fraction)
X_train_sampled = X_train_balanced[:sample_size]
y_train_sampled = y_train_balanced[:sample_size]
print(f"After random sampling (1/20), Training Set Shape: {X_train_sampled.shape}")

# Train other models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier()
}

performance_metrics = {}
for model_name, model in models.items():
    model.fit(X_train_sampled, y_train_sampled)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_proba)
    performance_metrics[model_name] = {
        'Accuracy': accuracy,
        'AUC': auc_score,
        'F1 Score': f1,
        'MCC': mcc,
        'Avg Precision': avg_precision
    }

# Display metrics
for model_name, metrics in performance_metrics.items():
    print(f"Performance for {model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 30)

# Bar plot
metrics_df = pd.DataFrame(performance_metrics)
metrics_df.plot(kind='bar', figsize=(12, 7))
plt.title('Comparison of Model Performance')
plt.ylabel('Score')
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ROC curve
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Precision-recall curve
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_score = auc(recall, precision)
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc_score:.2f})')
plt.title('Precision-Recall Curve for All Models')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# 读取正负样本数据
pos = pd.read_csv("pos.csv")
neg = pd.read_csv("neg.csv")

# 设置列索引范围
signal_cols = list(range(6, 16))   # 列6~15为电流特征
error_cols = list(range(16, 21))   # 列16~20为base-calling错误特征

# 设置有意义的特征名（请根据实际列定义进一步调整）
signal_feature_names = [
    'Current_Mean_Center',
    'Dwell_Time_Center',
    'Current_SD_Center',
    'Current_Mean_-2',
    'Current_Mean_-1',
    'Current_Mean_+1',
    'Current_SD_+1',
    'Dwell_Time_-1',
    'Dwell_Time_+1',
    'Dwell_Time_-2'
]

error_feature_names = [
    'Mismatch_Count',
    'Deletion_Count',
    'Insertion_Count',
    'Low_Quality_Base_Count',
    'Other_Error_Feature'  # 你可修改为具体含义
]

# 创建空列表用于保存p值
results = []

# 可视化和统计检验：电流特征
for i, col in enumerate(signal_cols):
    pos_val = pos.iloc[:, col]
    neg_val = neg.iloc[:, col]
    feature_name = signal_feature_names[i]
    
    # 检验
    p_val = mannwhitneyu(pos_val, neg_val, alternative='two-sided').pvalue
    results.append((feature_name, p_val))
    
    # 绘图
    df_plot = pd.DataFrame({
        'Value': pd.concat([pos_val, neg_val], axis=0),
        'Group': ['m7G'] * len(pos_val) + ['NormalG'] * len(neg_val)
    })
    
    plt.figure(figsize=(6, 5))
    sns.violinplot(x='Group', y='Value', data=df_plot, palette='Set2', inner='box')
    plt.title(f'{feature_name} - p={p_val:.4e}')
    plt.tight_layout()
    plt.show()

# 可视化和统计检验：basecalling特征
for i, col in enumerate(error_cols):
    pos_val = pos.iloc[:, col]
    neg_val = neg.iloc[:, col]
    feature_name = error_feature_names[i]
    
    # 检验
    p_val = mannwhitneyu(pos_val, neg_val, alternative='two-sided').pvalue
    results.append((feature_name, p_val))
    
    # 绘图
    df_plot = pd.DataFrame({
        'Value': pd.concat([pos_val, neg_val], axis=0),
        'Group': ['m7G'] * len(pos_val) + ['NormalG'] * len(neg_val)
    })
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(x='Group', y='Value', data=df_plot, palette='Set1')
    plt.title(f'{feature_name} - p={p_val:.4e}')
    plt.tight_layout()
    plt.show()

# 输出统计表格
df_results = pd.DataFrame(results, columns=['Feature', 'Mann-Whitney p-value'])
print(df_results)
