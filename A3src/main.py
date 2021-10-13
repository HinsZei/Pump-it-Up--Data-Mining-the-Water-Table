import Task1, Task7, Task2, Task3, Task4, Task6
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='usage description')
    parser.add_argument('id', help='task id such as task1(2,3,4,6,7)')
    parser.add_argument('method', help='method to run,options: train,test,predict')
    parser.add_argument('csv', help='full filename of the dataset to run')
    parser.add_argument('-n',
                        help='name of the neural network file(without suffix),optional,default:Models/Task?.pkl',
                        required=False, default=0)
    parser.add_argument('-t',
                        help='name of the Txt file in test method(without suffix),optional,default:Prediction/Task?.txt',
                        required=False, default=0)
    parser.add_argument('-c',
                        help='classifier of the Task 7, options: svc/rf , default: rf',
                        required=False, default='rf')
    args = parser.parse_args()
    try:
        id, method, csv, path, txt_path, classifier = args.id, args.method, args.csv, args.n, args.t, args.c
        methodDict = {
            'task1.train': Task1.train,
            'task1.test': Task1.test,
            'task1.predict': Task1.predict,
            'task2.train': Task2.train,
            'task2.test': Task2.test,
            'task2.predict': Task2.predict,
            'task3.train': Task3.train,
            'task3.test': Task3.test,
            'task3.predict': Task3.predict,
            'task4.train': Task4.train,
            'task4.test': Task4.test,
            'task4.predict': Task4.predict,
            'task6.train': Task6.train,
            'task6.test': Task6.test,
            'task6.predict': Task6.predict,
            'task7.train': Task7.train,
            'task7.test': Task7.test,
            'task7.predict': Task7.predict
        }
        if method == 'test':
            methodDict.get(id + '.' + method)(csv, path, txt_path)
        else:
            if id == 'task7' and method == 'train':
                methodDict.get(id + '.' + method)(csv, path, classifier)
            else:
                methodDict.get(id + '.' + method)(csv, path)
    except Exception as e:
        print(
            'usage:python main.py <task_id> <train|test|predict> <input_csv>  <-n NN_name>  <-t Txt_name>  <-c classifier>')
        print(e)
