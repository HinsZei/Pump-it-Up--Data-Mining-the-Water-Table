from Preprocessing import all_data, all_transformer, oversampling_Smote, oversampling_SmoteTomek, oversampling_ADASYN, \
    features_selection
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from collections import Counter


def train(csv,path):
    if path == 0:
        path = 'Models/Task4.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    X, y = all_data(csv, False)
    mlp = MLPClassifier(hidden_layer_sizes=[50, 50], alpha=0.001, max_iter=500, verbose=True, early_stopping=False,
                        learning_rate_init=5e-3, activation='relu')
    # add resampling method in the final model
    model = Pipeline([
        ('all_transformer', all_transformer()),
        # ('feature_selector',features_selection()), abandoned, score didn't improve
        ('oversampling', oversampling_SmoteTomek()),
        ('mlp', mlp)
    ])
    model.fit(X, y)
    dump(model, path)

'''same as task 2'''
def test(csv,path,txt_path):
    if path == 0:
        path = 'Models/Task4.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    if txt_path == 0:
        txt_path = 'Prediction/Task4.txt'
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
    print(Counter(y))
    print(Counter(y_predict))
    print(classification_report(y, y_predict, target_names=['functional needs repair', 'others']))

'''same as task 2'''
def predict(csv,path):
    if path == 0:
        path = 'Models/Task4.pkl'
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
