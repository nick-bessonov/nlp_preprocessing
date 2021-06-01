import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import ast


def main(train=None, test=None, train_nm_target=None, test_nm_target=None, train_text_field=None,
         test_text_field=None, n_tokens=None, params=None):  # tf_idf преобразование трейна и теста

    results = dict()

    if test is None:
        test = pd.DataFrame(data=np.zeros(len(train.columns)).reshape(1, len(train.columns)), columns=train.columns)
        test_text_field = train_text_field
        test_nm_target = train_nm_target

    y_train = train[train_nm_target]
    y_test = test[test_nm_target]

    if params is None:
        tf_idf_vectorizer = TfidfVectorizer()
    else:
        params = ast.literal_eval(params)
        tf_idf_vectorizer = TfidfVectorizer(**params)

    train_tf_idf_vector = tf_idf_vectorizer.fit_transform(train[train_text_field].values.astype('str'))
    test_tf_idf_vector = tf_idf_vectorizer.transform(test[test_text_field].values.astype('str'))

    # преобразуем feature name в цифры для обучения, так как русские символы не воспринимаются
    number_feature_list = [i for i in range(len(tf_idf_vectorizer.get_feature_names()))]
    # для скоров
    feature_tokens = tf_idf_vectorizer.get_feature_names()

    train_tf_idf_df = pd.DataFrame(data=train_tf_idf_vector.toarray(), columns=number_feature_list, index=train.index)
    test_tf_idf_df = pd.DataFrame(data=test_tf_idf_vector.toarray(), columns=number_feature_list, index=test.index)

    # функция подсчета среднего скора
    def mean_tf_idf(df, feature_tokens, n_tokens):

        mean_series = df.mean().sort_values(ascending=False).head(n_tokens)
        mean_series = mean_series.apply(lambda x: round(x, 3))

        top_features = [feature_tokens[idx] for idx in mean_series.index]
        top_df = pd.DataFrame(data=top_features, columns=['top features'], index=mean_series.index)
        top_df['score'] = mean_series

        return top_df

    # выводим tfidf скор в среднем по каждому токену на трейне и тесте
    train_score = mean_tf_idf(train_tf_idf_df, feature_tokens, n_tokens)
    test_score = mean_tf_idf(test_tf_idf_df, feature_tokens, n_tokens)

    train_tf_idf_df = pd.concat([train_tf_idf_df, y_train], axis=1)
    test_tf_idf_df = pd.concat([test_tf_idf_df, y_test], axis=1)

    results['output_train'] = train_tf_idf_df
    results['output_test'] = test_tf_idf_df
    results['output_vectorizer'] = tf_idf_vectorizer
    results['output_table_train'] = {'table_data': train_score.to_dict('records')}
    results['output_table_test'] = {'table_data': test_score.to_dict('records')}

    return results
