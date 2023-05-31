import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def main():
    # credit: read -> 0.7 -> switch -> ML-KNN -> res
    # read
    df = pd.read_csv('credit_train_land_credit_demo_phase4.csv')
    print(df.isna().mean())
    # 0.7
    df = df.loc[:, df.isnull().mean() < 0.7]
    # switch
    df.replace({'sex': {'男': 1, '女': 0}}, inplace=True)
    df.replace({'marriage': {'未婚': 0, '初婚': 1, '已婚': 2, '离婚': 3, '丧偶': 4, '未说': 5}}, inplace=True)
    df.replace(
        {'education': {'初中': 0, '中专': 1, '大专': 2, '技术学校': 3, '高中': 4, '大学本科': 5, '研究生': 6, '未知': 7,
                       '其他': 8}}, inplace=True)
    df.replace({'is_black': {'Y': 1, 'N': 0}}, inplace=True)
    df.replace({'is_contact': {'Y': 1, 'N': 0}}, inplace=True)
    # ML-KNN
    missing_cols = ['all_bal', 'bad_bal', 'due_intr', 'norm_bal', 'delay_bal']
    for target_col in missing_cols:
        feature_cols = ['credit_level', 'sex', 'marriage', 'education', 'is_black', 'is_contact']
        x = df.dropna()[feature_cols]
        y = df[target_col].dropna()
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(x, y)
        missing_rows = df[df[target_col].isnull()]
        predicted_values = knn_model.predict(missing_rows[feature_cols])
        df.loc[df[target_col].isnull(), target_col] = predicted_values
    # res
    print(df.isna().mean())
    print(df.dtypes)
    print(df.columns)
    df.drop('uid', axis=1, inplace=True)
    df.to_csv('credit_train_land_credit_demo_phase5.csv')

    # star: read -> 0.7 -> drop -> switch -> MEAN etc.
    # read
    df = pd.read_csv('credit_train_land_star_demo_phase4.csv')
    print(df.isna().mean())
    # 0.7
    df = df.loc[:, df.isnull().mean() < 0.7]
    # drop
    columns = ['avg_qur', 'oth_td_bal', 'td_crd_bal']
    df = df.drop(columns, axis=1)
    # switch
    df.replace({'sex': {'男': 1, '女': 0}}, inplace=True)
    df.replace({'marriage': {'未婚': 0, '初婚': 1, '已婚': 2, '离婚': 3, '丧偶': 4, '未说': 5, '再婚': 6}}, inplace=True)
    df.replace(
        {'education': {'初中': 0, '中专': 1, '大专': 2, '技术学校': 3, '高中': 4, '大学本科': 5, '研究生': 6, '未知': 7,
                       '其他': 8}}, inplace=True)
    df.replace({'is_black': {'Y': 1, 'N': 0}}, inplace=True)
    df.replace({'is_contact': {'Y': 1, 'N': 0}}, inplace=True)
    # MEAN etc.
    df['sex'].fillna(df['sex'].mode().iloc[0], inplace=True)
    df['marriage'].fillna(df['marriage'].mode().iloc[0], inplace=True)
    df['education'].fillna(df['education'].mode().iloc[0], inplace=True)
    df['is_black'].fillna(df['is_black'].mode().iloc[0], inplace=True)
    df['is_contact'].fillna(df['is_contact'].mode().iloc[0], inplace=True)
    columns = ['avg_mth', 'avg_year', 'sa_bal', 'td_bal', 'fin_bal', 'sa_crd_bal', 'sa_td_bal', 'ntc_bal', 'td_3m_bal',
               'td_6m_bal', 'td_1y_bal', 'td_2y_bal', 'td_3y_bal', 'td_5y_bal', 'cd_bal', 'all_bal']
    for column in columns:
        df[column].fillna(df[column].median(), inplace=True)
    # res
    print(df.isna().mean())
    print(df.dtypes)
    print(df.columns)
    df.drop('uid', axis=1, inplace=True)
    df.to_csv('credit_train_land_star_demo_phase5.csv')


if __name__ == '__main__':
    main()
