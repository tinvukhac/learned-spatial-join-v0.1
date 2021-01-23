import glob
import os
import pandas as pd


def main():
    print ('Concatenate tabular data')
    os.system('rm ../data/tabular/tabular_all.csv')
    tabular_data_files = glob.glob("../data/tabular/*.csv")
    df = pd.concat((pd.read_csv(f, header=0) for f in tabular_data_files))
    df.to_csv('../data/tabular/tabular_all.csv', index=0)


if __name__ == '__main__':
    main()
