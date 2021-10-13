from Preprocessing import all_transformer, oversampling_SmoteTomek, multiple_class_data
from joblib import dump, load
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from collections import Counter
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier


def train(csv, path, classifier):
    if path == 0:
        path = 'Models/Task7.pkl'
    else:
        path = 'Models/' + path + '.pkl'

    class_weight = {0: 0.1, 1: 0.85, 2: 0.05}
    # class_weight = 'balanced'
    if classifier == 'svc':
        ''' the performance of svc is poor with about 40% recall and 20% precision'''
        # m = SVC(verbose=True, class_weight=class_weight, tol=1e-3)
        m = SVC(verbose=True,class_weight='balanced',tol=1e-3)
    else:
        '''I don't think the cost-sensitive learning can be simply implemented by class_weight
            because I cannot specify the weights that need repair to functional separately
        '''
        m = BalancedRandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1,
                                           class_weight=class_weight, verbose=True)
    X, y = multiple_class_data(csv, False)
    model = Pipeline([
        ('all_transformer', all_transformer()),
        # ('feature_selector',features_selection()),
        # ('oversampling', oversampling_SmoteTomek()),
        ('classifier', m)
    ])
    model.fit(X, y)
    dump(model, path)


def test(csv, path, txt_path):
    if path == 0:
        path = 'Models/Task7.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    if txt_path == 0:
        txt_path = 'Prediction/Task7.txt'
    else:
        txt_path = 'Prediction/' + txt_path
    model = load(path)
    X, y = multiple_class_data(csv, False)
    y_predict = model.predict(X)
    with open(txt_path, 'w') as f:
        # count = 0
        for label in y_predict:
            if label == 0:
                f.write('functional\n')
            if label == 1:
                f.write('functional needs repair\n')
            if label == 2:
                f.write('non functional\n')
            # f.write(str(label) + ' ')
            # count += 1
            # if count % 30 == 0:
            #     f.write('\n')
    print(Counter(y))
    print(Counter(y_predict))
    print(classification_report(y, y_predict, target_names=['functional', 'functional needs repair', 'not functional']))


def predict(csv, path):
    if path == 0:
        path = 'Models/Task7.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    model = load(path)
    X = multiple_class_data(csv, True)
    y_predict = model.predict(X)
    X['label'] = y_predict
    index = X[X['label'] == 0].index.tolist()
    count = 0
    for id in index:
        print(id, end=' ')
        count += 1
        if count % 25 == 0:
            print()

