from optparse import OptionParser

from linear_regression_model import LinearRegressionModel


def main():
    print('Training the join cardinality estimator')

    parser = OptionParser()
    parser.add_option('-t', '--tab', type='string', help='Path to the tabular data file(CSV)')
    parser.add_option('-g', '--hist', type='string', help='Path to the histograms of input datasets')
    parser.add_option('-r', '--result', type='string', help='Path to the join result (CSV)')
    parser.add_option('-m', '--model', type='string', help='Path to the model to be saved')
    parser.add_option('-w', '--weights', type='string', help='Path to the model weights to be saved')
    parser.add_option('--train', action="store_true", dest="train", default=True)
    parser.add_option('--no-train', action="store_false", dest="train")

    (options, args) = parser.parse_args()
    options_dict = vars(options)

    try:
        tabular_path = options_dict['tab']
        histogram_path = options_dict['hist']
        join_result_path = options_dict['result']
        model_path = options_dict['model']
        model_weights_path = options_dict['weights']
        is_train = options_dict['train']
        lr = LinearRegressionModel()
        if is_train:
            lr.train(tabular_path, join_result_path, model_path)
        else:
            mse, mape, msle, mae = lr.test(tabular_path, join_result_path, model_path)
            print('mse: {}\nmape: {}\nmlse: {}\nmae: {}'.format(mse, mape, msle, mae))

    except RuntimeError:
        print('Please check your arguments')


if __name__ == "__main__":
    main()
