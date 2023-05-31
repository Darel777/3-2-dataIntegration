import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('credit_train_land_credit_demo_phase5.csv')

    # 处理 bad_bal 字段
    df.loc[df['bad_bal'] != 0, 'bad_bal'] = df['all_bal']

    # 处理 due_intr 字段
    mask = df['due_intr'] > df['bad_bal']
    rows = df.loc[mask].index
    df.loc[rows, 'due_intr'] = np.random.uniform(0, df.loc[mask, 'bad_bal'])

    # 处理 norm_bal 字段
    mask = (df['bad_bal'] != 0) & (df['norm_bal'] != 0) & (df['norm_bal'] != df['bad_bal'])
    rows = df.loc[mask].index
    df.loc[rows, 'norm_bal'] = df.loc[mask, 'bad_bal']

    # 处理 delay_bal 字段
    mask = df['delay_bal'] > df['bad_bal']
    rows = df.loc[mask].index
    df.loc[rows, 'delay_bal'] = np.random.uniform(0, df.loc[mask, 'bad_bal'])

    df.to_csv('credit_train_land_credit_demo_phase6.csv')


if __name__ == '__main__':
    main()
