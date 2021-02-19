from optparse import OptionParser

from linear_regression_model import LinearRegressionModel


def main():
    parser = OptionParser()
    parser.add_option('-m', '--model', type='string', help='Model name: {linear, dnn}')
    parser.add_option('-t', '--tab', type='string', help='Path to the tabular data file(CSV)')
    parser.add_option('-g', '--hist', type='string', help='Path to the histograms of input datasets')
    parser.add_option('-r', '--result', type='string', help='Path to the join result (CSV)')
    parser.add_option('-p', '--path', type='string', help='Path to the model to be saved')
    parser.add_option('-w', '--weights', type='string', help='Path to the model weights to be saved')
    parser.add_option('--train', action="store_true", dest="train", default=True)
    parser.add_option('--no-train', action="store_false", dest="train")

    (options, args) = parser.parse_args()
    options_dict = vars(options)

    model_names = ['linear', 'dnn']

    try:
        model_name = options_dict['model']
        if model_name not in model_names:
            print('Available model is {}'.format(','.join(model_names)))
        else:
            if model_name == 'linear':
                model = LinearRegressionModel()
            elif model_name == 'dnn':
                model = None

        tabular_path = options_dict['tab']
        histogram_path = options_dict['hist']
        join_result_path = options_dict['result']
        model_path = options_dict['path']
        model_weights_path = options_dict['weights']
        is_train = options_dict['train']

        if is_train:
            model.train(tabular_path, join_result_path, model_path, model_weights_path, histogram_path)
        else:
            mse, mape, msle, mae = model.test(tabular_path, join_result_path, model_path, model_weights_path, histogram_path)
            print('mse: {}\nmape: {}\nmlse: {}\nmae: {}'.format(mse, mape, msle, mae))

    except RuntimeError:
        print('Please check your arguments')


if __name__ == "__main__":
    main()
