from Preprocessing import all_data, all_transformer
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def train(csv,path):
    if path == 0:
        path = 'Models/Task2.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    mlp = MLPClassifier(hidden_layer_sizes=[50, 50], alpha=0.01, max_iter=500, verbose=True, early_stopping=False,
                        learning_rate_init=5e-3, activation='relu')
    X, y = all_data(csv, False)
    ''' if I save the MlpClassifier separately instead of including preprocessing, I may find that 
        when I run test() the model raise exception that 47 features needed but 46 features are given
        because once I fit_transform() the dataset , the preprocessing model will change. the test dataset
        lacks 1 column after one hot encoding, so I should save the preprocessing as well.
    '''
    model = Pipeline([
        ('all_transformer', all_transformer()),
        ('mlp', mlp)
    ])
    model.fit(X, y)
    dump(model, path)

'''same as task 1'''
def test(csv,path, txt_path):
    if path == 0:
        path = 'Models/Task2.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    if txt_path == 0:
        txt_path = 'Prediction/Task2.txt'
    else:
        txt_path = 'Prediction/' + txt_path
    model = load(path)
    X, y = all_data(csv, False)
    y_predict = model.predict(X)
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
    print(classification_report(y, y_predict, target_names=['functional needs repair', 'others']))

'''same as task 1'''
def predict(csv,path):
    if path == 0:
        path = 'Models/Task2.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    model = load(path)
    X = all_data(csv, True)
    y_predict = model.predict(X)
    X['label'] = y_predict
    index = X[X['label'] == 0].index.tolist()
    count = 0
    for id in index:
        print(id, end=' ')
        count += 1
        if count % 25 == 0:
            print()
