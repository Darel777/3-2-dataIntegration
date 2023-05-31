import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def main():
    cate_list = ['sex', 'marriage', 'education', 'is_black', 'is_contact']
    co_list1 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'bad_bal', 'due_intr', 'norm_bal', 'delay_bal']
    co_list2 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'avg_mth', 'avg_year', 'sa_bal', 'td_bal', 'fin_bal', 'sa_crd_bal', 'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal', 'td_1y_bal', 'td_2y_bal', 'td_3y_bal', 'td_5y_bal', 'cd_bal']
    df1 = pd.read_csv('credit_train_land_credit_demo_phase7.csv')
    df2 = pd.read_csv('credit_train_land_star_demo_phase7.csv')
    df1 = df1.iloc[:, 1:]
    df2 = df2.iloc[:, 1:]
    le = LabelEncoder()
    df1[cate_list] = df1[cate_list].apply(le.fit_transform)
    df2[cate_list] = df2[cate_list].apply(le.fit_transform)
    df1.to_csv('credit_trans.csv', index=False)
    df2.to_csv('star_trans.csv', index=False)
    scaler = StandardScaler()
    df1[co_list1] = scaler.fit_transform(df1[co_list1])
    df2[co_list2] = scaler.fit_transform(df2[co_list2])
    df1.to_csv('credit_std.csv', index=False)
    df2.to_csv('star_std.csv', index=False)


if __name__ == '__main__':
    main()
