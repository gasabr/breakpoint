import pandas as pd
from datetime import datetime

train_payments_file = 'data/qiwi_payments_data_train.csv'
train_users_file = 'data/qiwi_users_data_train.csv'
# test files
test_payments_file = 'data/qiwi_payments_data_test.csv'
test_users_file = 'data/qiwi_users_data_test.csv'


def load_train_set():
    train_payments = pd.DataFrame.from_csv(train_payments_file, sep=';')

    a = pd.read_csv(train_users_file, sep=';', parse_dates=[1])
    train_users = pd.DataFrame(a)

    train = pd.merge(train_users, train_payments, on='user_id')

    train = parse_dates(train)

    return train

def load_test_set():
    test_payments = pd.DataFrame.from_csv(test_payments_file, sep=';')

    a = pd.read_csv(test_users_file, sep=';', parse_dates=[1])
    test_users = pd.DataFrame(a)

    test = pd.merge(test_users, test_payments, on='user_id')

    test = parse_dates(test)

    return test


def parse_dates(dataset):
    ''' Parses `date_month` to unix time.

        One of the columns contain year and another month.
    '''

    dates = [datetime.strptime(dt, '%Y-%m') for dt in dataset['date_month']]

    unix = [int(dt.timestamp()) for dt in dates]

    dataset['unix_time'] = unix

    dataset.drop(['date_month'], axis=1)

    return dataset


def prepare_data():
    ''' Workflow to clean data. '''

    train = load_train_set()
    test  = load_test_set()

    test.to_csv('data/test.csv', index=False)
    train.to_csv('data/train.csv', index=False)


if __name__ == '__main__':

    train = pd.DataFrame(pd.read_csv('data/train.csv'))
    test  = pd.DataFrame(pd.read_csv('data/test.csv'))

    print(train.shape)
    print(test.shape)


