import pandas as pd


def main():
    print('Baseline: On Spatial Joins in MapReduce')

    # Computer # of block for Ahmed's large uniform datasets
    # block_df = pd.read_csv('data/temp/dataset.csv', delimiter='\t', header=0)
    # tabular_df = pd.read_csv('data/train_and_test_all_features/join_results_combined.csv', delimiter='\\s*,\\s*', header=0)
    # tabular_df = pd.merge(tabular_df, block_df, how='left', left_on=['dataset1', 'dataset2'], right_on=['dataset1', 'dataset2'])
    # tabular_df.to_csv('data/train_and_test_all_features/join_results_combined_updated_blocks.csv')

    


if __name__ == '__main__':
    main()
