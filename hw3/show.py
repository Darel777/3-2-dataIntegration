import pandas as pd


def main():
    star_train = pd.read_csv('credit_train_land_star_demo_phase2.csv')
    star_train.describe().to_csv('star_train_describe.csv')
    credit_train = pd.read_csv('credit_train_land_credit_demo_phase2.csv')
    credit_train.describe().to_csv('credit_train_describe.csv')


if __name__ == '__main__':
    main()
