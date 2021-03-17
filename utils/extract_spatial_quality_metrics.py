from utils import spatial_quality_extractor


def main():
    print('Extract spatial quality metrics')
    scales = ['large_aws', 'medium', 'small', 'real']
    master_paths = ['../data/spatial_quality_metrics/masters/large_aws',
                    '../data/spatial_quality_metrics/masters/medium',
                    '../data/spatial_quality_metrics/masters/small',
                    '../data/spatial_quality_metrics/masters/real']
    dataset_filenames_paths = ['../data/dataset_filenames/large_aws_filenames.csv',
                               '../data/dataset_filenames/medium_filenames.csv',
                               '../data/dataset_filenames/small_filenames_no_bit.csv',
                               '../data/dataset_filenames/real_filenames.csv']
    # Block size in MB
    block_sizes = [32, 1, 128.0 / 1024.0, 32]

    for master_path, dataset_filenames_path, scale, block_size in zip(master_paths, dataset_filenames_paths, scales, block_sizes):
        output_f = open('../data/spatial_quality_metrics/quality_metrics_{}.csv'.format(scale), 'w')
        output_f.writelines('dataset_name,block_size,total_area,total_margin,total_overlap,size_std,block_util\n')

        f = open(dataset_filenames_path)
        filenames = f.readlines()
        filenames = [f.strip() for f in filenames]

        gindex = 'zcurve' if scale == 'real' else 'rsgrove'
        extension = '.csv' if scale == 'large_aws' else ''

        for filename in filenames:
            master_file = '{}/{}/_master.{}'.format(master_path, filename, gindex)
            print(master_file)
            partitions = spatial_quality_extractor.get_partitions(master_file, block_size)
            total_area = spatial_quality_extractor.get_total_area(partitions)
            total_margin = spatial_quality_extractor.get_total_margin(partitions)
            total_overlap = spatial_quality_extractor.get_total_overlap(partitions)
            size_std = spatial_quality_extractor.get_size_std(partitions)
            block_util = spatial_quality_extractor.get_disk_util(partitions, block_size)
            dataset_filename = '{}/{}{}'.format(scale, filename, extension)
            output_f.writelines('{},{},{},{},{},{},{}\n'.format(dataset_filename, block_size * 1024 * 1024, total_area, total_margin, total_overlap, size_std, block_util))


if __name__ == '__main__':
    main()
