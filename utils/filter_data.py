def main():
    print ('Filter data')
    input_f = open('../data/train_and_test_all_features_split/train_join_results_small_x_small_pbsm.csv')
    output_f = open('../data/train_and_test_all_features_split/train_join_results_small_x_small_uniform_pbsm.csv', 'w')

    line = input_f.readline()
    distributions = ['gaussian', 'uniform']

    output_f.writelines(line)
    line = input_f.readline()

    while line:
        data = line.strip().split(',')
        # result_size = int(data[2])
        write = False
        # if 'diagonal' in data[0].lower() and 'gaussian' in data[1].lower():
        #     write = True
        if 'diagonal' in data[0].lower() and 'uniform' in data[1].lower():
            write = True
        # if 'uniform' in data[0].lower() and 'diagonal' in data[1].lower():
        #     write = True
        # if 'uniform' in data[0].lower() and 'uniform' in data[1].lower():
        #     write = True

        # join_sel = float(data[36])
        # min_sel = pow(10, -6)
        # max_sel = pow(10, -4)
        # if min_sel < join_sel < max_sel:
        #     write = True

        if write:
            output_f.writelines(line)

        line = input_f.readline()

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
