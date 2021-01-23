def main():
    print ('Filter data')
    input_f = open('../data/join_results/train/join_results_small_x_small.csv')
    output_f = open('../data/join_results/train/join_results_small_x_small_uniform.csv', 'w')

    lines = input_f.readlines()

    for line in lines:
        data = line.strip().split(',')
        # result_size = int(data[2])
        if 'uniform' in data[0].lower() or 'uniform' in data[1].lower():
            output_f.writelines(line)

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
