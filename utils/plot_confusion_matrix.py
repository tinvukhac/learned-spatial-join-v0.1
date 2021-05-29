import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    print('Plot confusion matrix')

    # Remove empty lines from Alberto's data
    # f = open('data/temp/theoretical_algorithm_selection.csv')
    # output_f = open('data/temp/algorithm_selection_b3.csv', 'w')
    #
    # lines = f.readlines()
    #
    # for line in lines:
    #     if len(line.strip()) > 0:
    #         output_f.writelines('{}\n'.format(line.strip()))
    #
    # output_f.close()
    # f.close()


    # Plot confusion matrix
    df = pd.read_csv('../data/temp/algorithm_selection_b3.csv', header=0)
    y_test = df['y_test']
    y_pred = df['y_pred']
    # cm = confusion_matrix(y_test, y_pred)

    class_names = ['BNLJ', 'PBSM', 'DJ', 'RepJ']
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4], normalize='true')
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Predicted algorithm', fontsize=16)
    plt.ylabel('Actual best algorithm', fontsize=16)
    plt.savefig('figures/confusion_matrix_with_normalization_b3.png')


if __name__ == '__main__':
    main()
