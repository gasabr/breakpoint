import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier

train_payments_file = 'data/qiwi_payments_data_train.csv'
train_users_file = 'data/qiwi_users_data_train.csv'
# test files
test_payments_file = 'data/qiwi_payments_data_test.csv'
test_users_file = 'data/qiwi_users_data_test.csv'


def load_train_set():
    train_payments = pd.read_csv(train_payments_file, sep=';')
    train_users    = pd.read_csv(train_users_file, sep=';')

    # merge users and payments by user_id
    train = pd.merge(train_users, train_payments, on='user_id')
    # parse dates
    train = parse_dates(train)

    return train

def load_test_set():
    test_payments = pd.read_csv(test_payments_file, sep=';')
    test_users    = pd.read_csv(test_users_file, sep=';')

    # merge users and payments by user_id
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

    dataset = dataset.drop(['date_month'], axis=1)

    return dataset


def prepare_data():
    ''' Workflow to clean data. 

        This function cleans the data, add it at the beggining of main()
        to refresh datasets. 
    '''

    # load the data from source
    train = load_train_set()
    test  = load_test_set()

    # encode sex in train set
    le = LabelEncoder()
    le.fit(train['sex'])
    train['sex'] = le.transform(train['sex'])

    # encode universities with labels
    train, test = binarize_uni(train, test)

    train, test = binarize_category(train, test)
    
    # fill None in graduation year with 0s
    train['graduation_year'] = train['graduation_year'].fillna(value=0)
    test['graduation_year']  = test['graduation_year'].fillna(value=0)

    # change `graduation_year` type to int
    train['graduation_year'] = train['graduation_year'].astype(int)
    test['graduation_year']  = test['graduation_year'].astype(int)

    test.to_csv('data/test.csv', index=False)
    train.to_csv('data/train.csv', index=False)


def binarize_uni(train, test):
    unis = set(train['university'])
    # unit 2 sets of names to get all of them
    unis = pd.Series(list(unis.union(test['university'])))
    le = LabelEncoder()
    le.fit(unis)
    # transformation
    train['university'] = le.transform(train[['university']])
    test['university'] = le.transform(test['university'])
    unis_transformed = le.transform(unis)

    # encode `university` with OneHotEncoding
    train_uni_bin = label_binarize(train['university'], unis_transformed)
    test_uni_bin = label_binarize(test['university'], unis_transformed)

    train_uni_bin = pd.DataFrame(train_uni_bin)
    test_uni_bin = pd.DataFrame(test_uni_bin)

    # add binarized columns to DataFrames
    train = pd.concat([train, train_uni_bin], axis=1)
    test  = pd.concat([test, test_uni_bin], axis=1)

    # drop encoded columns
    train = train.drop(['university'], axis=1)
    test  = test.drop(['university'], axis=1)

    return train, test


def binarize_category(train, test):
    # binarize cathogory column
    categories = set(train['category'])
    categories = pd.Series(list(categories.union(test['category'])))
    le = LabelEncoder()
    le.fit(categories)
    cols = ['cat={}'.format(i) for i in range(len(categories))]
    train_cat_bin = pd.DataFrame(label_binarize(train['category'], categories),
                                 columns=cols)
    test_cat_bin  = pd.DataFrame(label_binarize(test['category'], categories), 
                                 columns=cols)

    train = pd.concat([train, train_cat_bin], axis=1)
    test  = pd.concat([test, test_cat_bin], axis=1)

    train = train.drop(['category'], axis=1)
    test  = test.drop(['category'], axis=1)

    return train, test


def create_result():
    result = pd.read_csv('data/result.csv')
    clean_result = result[['user_id', 'sex', 'age']]
    clean_result = clean_result.drop_duplicates(['user_id'])
    clean_result.to_csv('data/result_clean_.csv', index=False)


if __name__ == '__main__':
    prepare_data()

    

