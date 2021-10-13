from Preprocessing import numerical_data, num_transformer
from plot_learning_curve import plot_learning_curve
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
def train(csv,path):
    #set default value of the save path
    if path == 0:
        path = 'Models/Task1.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    mlp = MLPClassifier(hidden_layer_sizes=[50, 50], alpha=0.01, max_iter=500, verbose=True, early_stopping=False,
                        learning_rate_init=5e-3, activation='relu')
    # mlp = MLPClassifier(hidden_layer_sizes=[50, 50], alpha=0.01, max_iter=500, verbose=True, early_stopping=False,
    #                     learning_rate_init=5e-3, activation='tanh')
    X, y = numerical_data(csv, False)
    # build the model including preprocessing and neural network, if you don't do it like this, you may have issues in Task 2
    model = Pipeline([
        ('num_transformer', num_transformer()),
        ('mlp', mlp)
    ])
    # ploting the learning curve
    # plt = plot_learning_curve(model, 'learning curve:relu,not early stopping', X, y, )
    # plt.savefig('relu,not early stopping.png')
    # plt.show()
    model.fit(X, y)
    #save the model
    dump(model, path)


def test(csv,path,txt_path):
    # default path
    if path == 0:
        path = 'Models/Task1.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    if txt_path == 0:
        txt_path = 'Prediction/Task1.txt'
    else:
        txt_path = 'Prediction/' + txt_path
    model = load(path)
    X, y = numerical_data(csv, False)
    y_predict = model.predict(X)
    # convert 0,1 to the labels again
    with open(txt_path, 'w') as f:
        # count = 0
        for label in y_predict:
            if label == 0:
                f.write('functional needs repair\n')
            if label == 1:
                f.write('others\n')
            # f.write(str(label) + ' ')
            # count += 1
            # if count % 30 == 0:
            #     f.write('\n')
# give all the normal metrics for classification problem so that this part of code don't have to change in next tasks
    print(classification_report(y, y_predict, target_names=['functional needs repair', 'others']))


def predict(csv,path):
    if path == 0:
        path = 'Models/Task1.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    model = load(path)
    X = numerical_data(csv, True)
    y_predict = model.predict(X)
    #use original dataset to get the id(index)
    X['label'] = y_predict
    index = X[X['label'] == 0].index.tolist()
    count = 0
    for id in index:
        print(id, end=' ')
        count += 1
        if count % 25 == 0:
            print()