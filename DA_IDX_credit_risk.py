# Library python 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score,classification_report
from sklearn.metrics import RocCurveDisplay, roc_curve

# Load Data
df = pd.read_csv('D:/PBI/DA/loan_data_2007_2014.csv')
print("Dimensi Dataset : ",df.shape)
df.head()

# Preprocessing Data
print("Column Duplicate : ",df.duplicated().sum())
# Drop kolom NaN yang tidak dapat di analisis
df1 = df.dropna(axis=1, how='all')
df1 = df1.drop(['Unnamed: 0', 'url', 'desc', 'title', 'zip_code', 'id', 'member_id'], axis=1)

var_cat = df1.select_dtypes(include = 'object').columns # Kolom kategorik

# cek unique values dari tiap kolom kategorik
for kolom in var_cat:
    unique = df1[kolom].unique()
    print(f"Kolom : {kolom} \n {unique}\n\n")


df1 = df1.drop(['application_type'], axis=1) # Drop kolom application_type karena hanya memiliki 1 unique value
df2 = df1.copy()

# Format tanggal
date_format = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
df2[date_format] = df2[date_format].apply(lambda x: '01-' + x)

# Convert to datetime
for col in date_format:
    df2[col] = pd.to_datetime(df2[col], format='%d-%b-%y')

df2[date_format].head()

# Jumlah nilai unik untuk setiap kolom object
object_columns = df2.select_dtypes(include=['object']).columns
df2[object_columns].nunique()


# Drop kolom emp_title karena memiliki terlalu banyak unique value
df2 = df2.drop(['emp_title'], axis = 1)

object_columns = df2.select_dtypes(include=['object']).columns
unik_cat = {}

for col in object_columns:
    unik_cat[col] = df2[col].value_counts()

for col, counts in unik_cat.items():
    print(f"Jumlah Unique Value Pada Kolom '{col}':")
    print(counts)
    print()

# Memiliki nilai yang sama
df_p1 = df2.drop(['policy_code'], axis =1)

# Mengubah nilai loan_status menjadi 0 dan 1
df_p1['loan_status'] = df_p1['loan_status'].replace(['Current', 'Fully Paid', 'In Grace Period', 'Does not meet the credit policy. Status:Fully Paid'], 1).replace(['Charged Off', 'Late (31-120 days)', 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off'], 0)
df_p1['loan_status'].value_counts()

# Exploratory Data Analysis (EDA)

counts = df_p1['loan_status'].value_counts()

# Fungsi heuristik untuk menentukan apakah sebuah status termasuk 'baik'
def is_good_status(lbl):
    s = str(lbl).strip().lower()
    if s in ('1', '1.0'):
        return True
    good_keywords = ('fully paid', 'paid', 'current', 'completed', 'active')
    if any(k in s for k in good_keywords):
        return True
    return False

# Agregasi ke dua kategori: baik vs tidak baik
good_total = counts[[i for i in counts.index if is_good_status(i)]].sum() if any(is_good_status(i) for i in counts.index) else 0
bad_total = counts.sum() - good_total

sizes = [good_total, bad_total]
labels = ['Pinjaman Baik (1)', 'Pinjaman Tidak Baik (0)']
colors = ['#2ca02c', '#d62728']

fig, ax = plt.subplots(figsize=(10,10))
explode = (0.06, 0.0)  
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.78,
    explode=explode,
    colors=colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)

centre_circle = plt.Circle((0,0), 0.55, fc='white')
fig.gca().add_artist(centre_circle)

# Styling teks autopct dan judul
for t in autotexts:
    t.set_color('white')
    t.set_fontsize(11)
for t in texts:
    t.set_fontsize(11)

ax.set_title('Status Pinjaman', fontsize=14, pad=3)
ax.legend(wedges, [f'{labels[0]}: {good_total}', f'{labels[1]}: {bad_total}'],
          title='Jumlah', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

# ax.axis('equal')  # pastikan pie berbentuk lingkaran
plt.tight_layout()
plt.show()

counts = df_p1['emp_length'].value_counts()
order = [
    '< 1 year', '1 year', '2 years', '3 years', '4 years',
    '5 years', '6 years', '7 years', '8 years', '9 years',
    '10+ years'
]

counts = counts.reindex(order).dropna()

fig, ax = plt.subplots(figsize=(10, 6))

# Bar chart
bars = ax.bar(
    counts.index,
    counts.values,
    edgecolor='black',
    alpha=0.9
)

# Line chart 
ax.plot(
    counts.index,
    counts.values,
    marker='.',
    color='red',
    linewidth=4
)

ax.set_title('Distribusi Lama Bekerja (Employment Length)', fontsize=14, pad=8)
ax.set_xlabel('Lama Bekerja', fontsize=11)
ax.set_ylabel('Jumlah Peminjam', fontsize=11)
plt.xticks(rotation=45, ha='right')

total = counts.sum()
for i, value in enumerate(counts.values):
    ax.text(
        i,
        value,
        f'{int(value)}\n({value/total:.1%})',
        ha='center',
        va='bottom',
        fontsize=9
    )

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

counts = df_p1['grade'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))

# Bar chart
bars = ax.bar(
    counts.index,
    counts.values,
    edgecolor='black',
    alpha=0.85
)

ax.set_title('Distribusi Grade Pinjaman', fontsize=14, pad=2)
ax.set_xlabel('Grade', fontsize=11)
ax.set_ylabel('Jumlah Peminjam', fontsize=11)

total = counts.sum()
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}\n({height/total:.1%})',
        ha='center',
        va='bottom',
        fontsize=9
    )

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

counts = df_p1['home_ownership'].fillna('NA').astype(str).value_counts()
preferred_order = ['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE']

ordered = [c for c in preferred_order if c in counts.index]
others = [c for c in counts.index if c not in ordered]
counts = counts.reindex(ordered + others, fill_value=0)

fig, ax = plt.subplots(figsize=(10,5))
palette = plt.get_cmap('tab10').colors
bar_colors = palette[:len(counts)]
bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor='white', linewidth=0.7, alpha=0.95)

ax.set_title('Distribusi Kepemilikan Rumah (Home Ownership)', fontsize=14, pad=20)
ax.set_xlabel('Status Kepemilikan Rumah', fontsize=11)
ax.set_ylabel('Jumlah Peminjam', fontsize=11)
plt.xticks(rotation=30, ha='right')

total = counts.sum()
for rect in bars:
    h = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        h + total * 0.004,
        f'{int(h)}\n({h/total:.1%})',
        ha='center',
        va='bottom',
        fontsize=9
    )

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Modeling Data
data = df_p1.copy()

# Split data menjadi fitur dan target
X = data.drop('loan_status', axis =1)
y = data['loan_status']

X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.2, stratify=y, random_state=0))

y_train.value_counts()

class NumImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric = X.select_dtypes(include=['int64', 'float64']).columns
        X[numeric] = IterativeImputer().fit_transform(X[numeric])
        return X
class CatImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cat = X.select_dtypes(include=['object']).columns
        X[cat] = SimpleImputer(strategy = 'most_frequent').fit_transform(X[cat])
        return X

class scaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        num = X.select_dtypes(include=['int64', 'float64']).columns
        X[num] = MinMaxScaler().fit_transform(X[num])
        return X
class OrdinalEnc(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        oe = OrdinalEncoder(categories=[['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']])
        X['emp_length'] = oe.fit_transform(X[['emp_length']])
        return X

class Labelenc(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        # Inisialisasi LabelEncoder
        label_encoders = {}
        
        # Lakukan label encoding untuk setiap kolom kategorik
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        return X
    
pipe_prepo = Pipeline([
    ('num_imputer', NumImputer()),
    ('cat_imputer', CatImputer()),
    ('scaler', scaler()),
    ('ordinal encoder', OrdinalEnc()),
    ('label encoder', Labelenc())
])

X_train = pipe_prepo.fit_transform(X_train)
X_test = pipe_prepo.transform(X_test)

X_train.isna().sum().sum()
X_test.isna().sum().sum()

X_train = X_train.select_dtypes(exclude=['datetime64'])
X_test = X_test.select_dtypes(exclude=['datetime64'])

list_model = [
    LogisticRegression(random_state=0, max_iter=1000), 
    MultinomialNB(),
    KNeighborsClassifier()
]

result = []

for model in list_model:
    pipeline = Pipeline([
        ('classifier', model)
    ]) 

    try:
        pipeline.fit(X_train, y_train) 
        y_pred = pipeline.predict(X_test) 

        bal_accuracy = balanced_accuracy_score(y_test, y_pred) 
        f1 = f1_score(y_test, y_pred, average='binary') 
        
        if hasattr(pipeline, "predict_proba"):
            roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]) 
        else:
            roc_auc = None
        
        hasil = {
            'Model': type(model).__name__, 
            'Balanced Accuracy': bal_accuracy,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        }
        result.append(hasil)
    except Exception as e:
        print(f"Gagal melatih model {type(model).__name__}: {e}")

result_all = pd.DataFrame(result)
print(result_all)

best = Pipeline([
        ('classifier', LogisticRegression(random_state=0))
    ])
best.fit(X_train, y_train)


Y_pred = best.predict(X_test)

bal_accuracy = balanced_accuracy_score(y_test, Y_pred)
f1 = f1_score(y_test, Y_pred, average='binary')
roc_auc = roc_auc_score(y_test, best.predict_proba(X_test)[:, 1])

print(classification_report(y_test, Y_pred))
print(f'Balance Aquracy = {bal_accuracy}\nF1 score \t= {f1}\nROC AUC \t= {roc_auc}')

fig, ax = plt.subplots(figsize=(8, 7))

RocCurveDisplay.from_estimator(
    best, 
    X_test, 
    y_test, 
    ax=ax, 
    name='Logistic Regression', 
    color='darkorange',
    linewidth=2
)

ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()