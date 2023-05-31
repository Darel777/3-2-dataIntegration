import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


def main():
    df1 = pd.read_csv('credit_std.csv')
    co_list1 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'bad_bal', 'due_intr', 'norm_bal',
                'delay_bal']
    df1[co_list1].corr().to_csv('credit_corr.csv', index=False)
    df2 = pd.read_csv('star_std.csv')
    co_list2 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'avg_mth', 'avg_year', 'sa_bal',
                'td_bal', 'fin_bal', 'sa_crd_bal', 'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal', 'td_1y_bal',
                'td_2y_bal', 'td_3y_bal', 'td_5y_bal', 'cd_bal']
    df2[co_list2].corr().to_csv('star_corr.csv', index=False)

    df1 = df1.drop('norm_bal', axis=1)
    df2 = df2.drop('sa_crd_bal', axis=1).drop('td_3y_bal', axis=1).drop('td_5y_bal', axis=1)

    data = df1.iloc[:, [7, 9]]
    pca = PCA(n_components=1)
    data_pca = pca.fit_transform(data)
    df1['bad&delay'] = data_pca
    df1.drop(df1.columns[[7, 9]], axis=1, inplace=True)
    df1.to_csv('credit_after_pca.csv', index=False)

    data = df2.iloc[:, [6, 7, 8, 10, 18]]
    pca = PCA(n_components=0.99)
    data_pca = pca.fit_transform(data)
    df2['pca_1'] = data_pca[:, 0]
    df2['pca_2'] = data_pca[:, 1]
    df2['pca_3'] = data_pca[:, 2]
    df2.drop(df2.columns[[6, 7, 8, 10, 18]], axis=1, inplace=True)
    df2.to_csv('star_after_pca.csv', index=False)

    df1 = pd.read_csv('credit_after_pca.csv')
    co_list1 = ['credit_level', 'sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'due_intr',
                'bad&delay']
    df1[co_list1].corr().to_csv('credit_corr_after_pca_corr.csv')
    df2 = pd.read_csv('star_after_pca.csv')
    co_list2 = ['star_level', 'sex', 'marriage', 'education', 'is_black', 'is_contact', 'sa_bal', 'fin_bal',
                'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal', 'td_1y_bal', 'td_2y_bal', 'pca_1', 'pca_2', 'pca_3']
    df2[co_list2].corr().to_csv('star_corr_after_pca_corr.csv')

    y = df1['credit_level']
    x = df1.drop(['credit_level'], axis=1)

    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["features"] = x.columns
    print(vif)

    y = df2['star_level']
    x = df2.drop(['star_level'], axis=1)

    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif["features"] = x.columns
    print(vif)


if __name__ == '__main__':
    main()
