from Preprocessing import all_data, all_transformer, oversampling_SmoteTomek
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd


def train(csv,path):
    if path == 0:
        path = 'Models/Task6.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    X, y = all_data(csv, False)
    mlp = MLPClassifier(max_iter=1000, early_stopping=True, verbose=0)
    # total 72 combinations
    grid_param = {'hidden_layer_sizes': ([50, 50], [100, 100], [100]),
                  'alpha': [0.01, 0.0001],
                  'solver': ['adam', 'sgd'],
                  'learning_rate_init': [1e-3, 1e-2, 5e-3],
                  'tol': [1e-4, 1e-5],
                  # 'activation':['relu','tanh']
                  }
    '''My CPU is AMD 3900x which has 12 cores, even I used all the cores it cost me about 10 mins, if
        you gonna test this method please be patient XD
    '''
    grid_search = GridSearchCV(mlp, param_grid=grid_param, n_jobs=4, scoring='f1')
    model = Pipeline([
        ('all_transformer', all_transformer()),
        # ('feature_selector',features_selection()),
        ('oversampling', oversampling_SmoteTomek()),
        ('mlp', grid_search)
    ])
    model.fit(X, y)
    dump(model, path)


def test(csv,path, txt_path):
    if path == 0:
        path = 'Models/Task6.pkl'
    else:
        path = 'Models/' + path + '.pkl'
    if txt_path == 0:
        txt_path = 'Prediction/Task6.txt'
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
    ''' the model only uses the best one to predict'''
    print('The best found model outputs:')
    print(classification_report(y, y_predict, target_names=['functional needs repair', 'others']))
    # print('grid search outputs:')
    '''the ranking shows that
        lr of 1e-3 and 1e-2 might cause it fall into local minimal
        single layer with 100 units might be too simple and 100,100 might be useless
        sgd is quite slow
        
    '''
    result = pd.DataFrame.from_dict(model.steps[2][1].cv_results_)
    result.to_csv('Model_comparasion.csv')
    # print(result)


def predict(csv,path):
    if path == 0:
        path = 'Models/Task6.pkl'
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
