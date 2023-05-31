import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def star():
    df = pd.read_csv('credit_train_land_star_demo_phase2.csv')
    all_list = ['uid', 'star_level', 'sex', 'marriage', 'education', 'is_black', 'is_contact', 'sa_bal', 'fin_bal',
                'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal', 'td_1y_bal', 'td_2y_bal', 'all_bal', 'avg_mth',
                'avg_year', 'td_bal', 'cd_bal']
    df = df.loc[:, all_list]

    df['sex'].fillna(df['sex'].mode().iloc[0], inplace=True)
    df['marriage'].fillna(df['marriage'].mode().iloc[0], inplace=True)
    df['education'].fillna(df['education'].mode().iloc[0], inplace=True)
    df['is_black'].fillna(df['is_black'].mode().iloc[0], inplace=True)
    df['is_contact'].fillna(df['is_contact'].mode().iloc[0], inplace=True)
    columns = ['avg_mth', 'avg_year', 'sa_bal', 'td_bal', 'fin_bal', 'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal',
               'td_1y_bal', 'td_2y_bal', 'cd_bal', 'all_bal']
    for column in columns:
        df[column].fillna(df[column].median(), inplace=True)

    cate_list = ['sex', 'marriage', 'education', 'is_black', 'is_contact']
    co_list = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'sa_bal', 'fin_bal',
               'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal', 'td_1y_bal', 'td_2y_bal', 'all_bal', 'avg_mth',
               'avg_year', 'td_bal', 'cd_bal']
    le = LabelEncoder()
    df[cate_list] = df[cate_list].apply(le.fit_transform)
    scaler = StandardScaler()
    df[co_list] = scaler.fit_transform(df[co_list])

    data = df.iloc[:, [15, 16, 17, 18, 19]]
    pca = PCA(n_components=0.99)
    data_pca = pca.fit_transform(data)
    df['pca_1'] = data_pca[:, 0]
    df['pca_2'] = data_pca[:, 1]
    df['pca_3'] = data_pca[:, 2]
    df.drop(df.columns[[15, 16, 17, 18, 19]], axis=1, inplace=True)

    df_neg = df.loc[df['star_level'] == -1]
    df_pos = df.loc[df['star_level'] != -1]

    co_list1 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'sa_bal', 'fin_bal',
                'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal', 'td_1y_bal', 'td_2y_bal', 'pca_1', 'pca_2', 'pca_3']
    x_train = df_pos[co_list1]
    y_train = df_pos['star_level']
    x_test = df_neg[co_list1]

    lr = LogisticRegression(max_iter=500)  # 迭代到10000也只有0.02%的概率差
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    xgb = XGBClassifier()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    y_pred_xgb += 1
    w1 = 0.8278268360973509 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    w2 = 0.8828774499246177 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    w3 = 0.9021753176825329 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    w4 = 0.9066982554382942 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    y_pred_weighted = (w1 * y_pred_lr + w2 * y_pred_dt + w3 * y_pred_rf + w4 * y_pred_xgb)
    nearest_idx = np.abs(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) - y_pred_weighted.reshape(-1, 1)).argmin(axis=1)
    y_pred_weighted = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[nearest_idx]
    df_neg['star_level'] = y_pred_weighted

    df_neg.to_csv('result_star.csv')


def credit():
    df = pd.read_csv('credit_train_land_credit_demo_phase2.csv')
    all_list = ['uid', 'credit_level', 'sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'bad_bal', 'due_intr', 'delay_bal']
    df = df.loc[:, all_list]

    missing_cols = ['all_bal', 'bad_bal', 'due_intr', 'delay_bal']
    cate_list = ['sex', 'marriage', 'education', 'is_black', 'is_contact']
    co_list = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'bad_bal', 'due_intr', 'delay_bal']
    le = LabelEncoder()
    df[cate_list] = df[cate_list].apply(le.fit_transform)
    for target_col in missing_cols:
        feature_cols = ['credit_level', 'sex', 'marriage', 'education', 'is_black', 'is_contact']
        x = df.dropna()[feature_cols]
        y = df[target_col].dropna()
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(x, y)
        missing_rows = df[df[target_col].isnull()]
        predicted_values = knn_model.predict(missing_rows[feature_cols])
        df.loc[df[target_col].isnull(), target_col] = predicted_values

    df.loc[df['bad_bal'] != 0, 'bad_bal'] = df['all_bal']
    mask = df['due_intr'] > df['bad_bal']
    rows = df.loc[mask].index
    df.loc[rows, 'due_intr'] = np.random.uniform(0, df.loc[mask, 'bad_bal'])
    mask = df['delay_bal'] > df['bad_bal']
    rows = df.loc[mask].index
    df.loc[rows, 'delay_bal'] = np.random.uniform(0, df.loc[mask, 'bad_bal'])

    scaler = StandardScaler()
    df[co_list] = scaler.fit_transform(df[co_list])

    data = df.iloc[:, [8, 10]]
    pca = PCA(n_components=1)
    data_pca = pca.fit_transform(data)
    df['bad&delay'] = data_pca
    df.drop(df.columns[[8, 10]], axis=1, inplace=True)

    df_neg = df.loc[df['credit_level'] == -1]
    df_pos = df.loc[df['credit_level'] != -1]

    co_list1 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'due_intr', 'bad&delay']
    x_train = df_pos[co_list1]
    y_train = df_pos['credit_level']
    x_test = df_neg[co_list1]

    lr = LogisticRegression(max_iter=500)  # 迭代到10000也只有0.02%的概率差
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    xgb = XGBClassifier()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    y_pred_xgb += 1
    y_pred_xgb = np.where(y_pred_xgb == 1, 35, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 2, 50, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 3, 60, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 4, 70, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 5, 85, y_pred_xgb)
    w1 = 0.7286135693215339 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    w2 = 0.8964274008521796 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    w3 = 0.8970829236315963 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    w4 = 0.9062602425434284 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    y_pred_weighted = (w1 * y_pred_lr + w2 * y_pred_dt + w3 * y_pred_rf + w4 * y_pred_xgb)
    nearest_idx = np.abs(np.array([35, 50, 60, 70, 85]) - y_pred_weighted.reshape(-1, 1)).argmin(axis=1)
    y_pred_weighted = np.array([35, 50, 60, 70, 85])[nearest_idx]
    df_neg['credit_level'] = y_pred_weighted

    df_neg.to_csv('result_credit.csv')


if __name__ == '__main__':
    star()
    credit()
