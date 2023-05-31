# p2 2mysql
import pymysql as mysql
import pandas as pd


def main():
    df = pd.read_csv('存款汇总——去重.csv')
    df = df.fillna(value='None')
    mydb = mysql.connect(
        host="localhost",
        user="root",
        password="root",
        port=3306,
        database="credit_train_land"
    )

    cursor = mydb.cursor()

    for index, row in df.iterrows():
        sql = "INSERT INTO asset_info (uid, all_bal, avg_mth, avg_qur, avg_year, sa_bal, td_bal, fin_bal, sa_crd_bal, td_crd_bal, sa_td_bal, ntc_bal, td_3m_bal, td_6m_bal, td_1y_bal, td_2y_bal, td_3y_bal, td_5y_bal, oth_td_bal, cd_bal) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (row['uid'], row['all_bal'], row['avg_mth'], row['avg_qur'], row['avg_year'], row['sa_bal'], row['td_bal'], row['fin_bal'], row['sa_crd_bal'], row['td_crd_bal'], row['sa_td_bal'], row['ntc_bal'], row['td_3m_bal'], row['td_6m_bal'], row['td_1y_bal'], row['td_2y_bal'], row['td_3y_bal'], row['td_5y_bal'], row['oth_td_bal'], row['cd_bal'])
        cursor.execute(sql, val)

    mydb.commit()


if __name__ == '__main__':
    main()
