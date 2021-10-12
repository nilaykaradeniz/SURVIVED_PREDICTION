import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import  LabelEncoder
import os
pd.set_option('display.expand_frame_repr', False)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def grab_col_names(df, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def target_summary_with_num(df, target, num_col):
    print(df.groupby(target).agg({num_col: "mean"}), end="\n\n\n")

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show()

def missing_values_table(df, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def label_encoder(df, binary_col):
    labelencoder = LabelEncoder()
    df[binary_col] = labelencoder.fit_transform(df[binary_col])
    return df

def one_hot_encoder(df, categorical_cols, drop_first=False):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df

def rare_analyser(df, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(df[col].value_counts()))
        print(pd.DataFrame({"COUNT": df[col].value_counts(),
                            "RATIO": df[col].value_counts() / len(df),
                            "TARGET_MEAN": df.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(df, rare_perc, cat_cols):
    rare_columns = [col for col in cat_cols if (df[col].value_counts() / len(df) < 0.01).sum() > 1]
    for col in rare_columns:
        tmp = df[col].value_counts() / len(df)
        rare_labels = tmp[tmp < rare_perc].index
        df[col] = np.where(df[col].isin(rare_labels), 'Rare', df[col])
    return df

path=os.getcwd()
df = pd.read_csv(path +r"\source_file\titanic.csv")
df.describe().T
df.isnull().sum()
df.shape

def titanic_data_prep(df):
    df.columns = [col.upper() for col in df.columns]
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    df["NEW_NAME_COUNT"] = df["NAME"].str.len()
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    for col in num_cols:
        print(col, check_outlier(df, col))

    for col in num_cols:
        replace_with_thresholds(df, col)

    for col in num_cols:
        print(col, check_outlier(df, col))

    missing_values_table(df)
    df.drop("CABIN", inplace=True, axis=1)

    remove_cols = ["TICKET", "NAME"]
    df.drop(remove_cols, inplace=True, axis=1)
    df.head()

    missing_values_table(df)

    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = label_encoder(df, col)

    rare_analyser(df, "SURVIVED", cat_cols)
    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

    df = one_hot_encoder(df, ohe_cols)
    df.head()
    df.shape

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    rare_analyser(df, "SURVIVED", cat_cols)

    useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                    (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

    df.drop(useless_cols, axis=1, inplace=True)
    return df

df_prep =titanic_data_prep(df)
df_prep =df_prep.dropna()

cat_cols, num_cols, cat_but_car = grab_col_names(df_prep)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

y = df_prep["SURVIVED"]
X = df_prep.drop(["SURVIVED","PASSENGERID","NEW_TITLE"], axis=1)
#X.shape
def high_correlated_cols(df, plot=False, corr_th=0.85):
    corr = df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list
X=X.drop(high_correlated_cols(X),axis=1)
#X.shape

log_model = LogisticRegression( max_iter=1000).fit(X, y)
log_model.intercept_
log_model.coef_
y_pred = log_model.predict(X)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()
plot_confusion_matrix(y, y_pred)
print(classification_report(y, y_pred))
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=42)
log_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()
roc_auc_score(y_test, y_prob)


y = df_prep["SURVIVED"]
X = df_prep.drop(["SURVIVED","PASSENGERID","NEW_TITLE"], axis=1)
log_model = LogisticRegression(max_iter=10000).fit(X, y)
cv_results = cross_validate(log_model,X, y, cv=5,scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_precision'].mean()
cv_results['test_recall'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()





















