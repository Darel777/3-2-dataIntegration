import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    df = pd.read_csv('credit_train_land_credit_demo_phase3.csv')

    for col in df.columns:
        # 去除NaN
        col_data = df[col].dropna()

        # 去除无穷量
        if np.isinf(col_data).sum() > 0:
            col_data = col_data.replace([np.inf, -np.inf], np.nan).dropna()

        # 分别计算频数，分箱
        if col == 'five_class':
            freq, bins = np.histogram(col_data, bins=[0, 1, 2, 3, 4, 5, 6, 7])
        else:
            continue

        # 绘制直方图
        plt.bar(range(len(freq)), freq)

        # 添加标注
        for i, v in enumerate(freq):
            plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')

        # 设置X轴标签
        plt.xticks(range(len(bins) - 1), bins[:-1])

        plt.title('Histogram of {}'.format(col))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig('credit_hist_{}.png'.format(col))


if __name__ == '__main__':
    main()
