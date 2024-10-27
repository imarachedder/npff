import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.stats import mode
import joblib
from preproccesing import preprocess_data

def format_predictions(predictions_df):
    """
        Функция форматирования предсказанных данных в читаемый
    :param predictions_df: Датафрейм с предсказанными данными
    :return:
    """
    # Преобразование второй колонки из 0/1 в "Нет"/"Да"
    predictions_df['Ранний выход на пенсию'] = predictions_df['Ранний выход на пенсию'].apply(lambda x: 'Да' if x == 1 else 'Нет')

    # Преобразование третьей колонки в зависимости от значения
    def format_retirement_status(years):
        if years < 0:
            return f"{abs(years)} лет НА пенсии"
        elif years == 0:
            return "Вышел на пенсию"
        else:
            return f"{years} лет ДО пенсии"
    
    predictions_df['Пенсионный статус'] = predictions_df['Пенсионный статус'].apply(format_retirement_status)

    return predictions_df

def test_2(data_cleaned, target_column='erly_pnsn_flg', id_column='accnt_id'):
    """
        Функция выполняет предсказания и возвращает Датафрей.
    :param data_cleaned: обработанные данные
    :param target_column: целевая переменная
    :param id_column: ID Клиента
    :return: предсказанные данные
    """
    # Загрузка обученных моделей
    isolation_forest = joblib.load('./model/isolation_forest_model.pkl')
    elliptic_envelope = joblib.load('./model/elliptic_envelope_model.pkl')

    # Подготовка данных для тестирования (исключаем ID и целевую переменную)
    print(data_cleaned.columns)
    data_copy = data_cleaned.copy()
    print(data_copy.columns)
    X_test = data_copy.drop(columns=['clnt_id', id_column, 'erly_pnsn_flg', 'age_retirement'])

    # Предсказания каждой модели
    y_pred_if = isolation_forest.predict(X_test)
    y_pred_ee = elliptic_envelope.predict(X_test)

    # Ансамблевое предсказание (голосование)
    predictions = np.vstack((y_pred_if, y_pred_ee)).T
    y_pred = mode(predictions, axis=1)[0].flatten()
    y_pred_final = np.where(y_pred == 1, 0, 1)  # Преобразуем к значениям 0 и 1

    # Создание DataFrame с предсказаниями
    predictions_df = pd.DataFrame({
        id_column: data_cleaned[id_column],
        'erly_pnsn_flg': y_pred_final
    })

    predictions_df.to_csv('test_predictions_with_ids.csv', index=False, sep=',', encoding='utf-8')

    return predictions_df

def test_3(data_cleaned, target_column='erly_pnsn_flg', id_column='accnt_id'):
    """
        Функция выполняет предсказания и возвращает Датафрей.
    :param data_cleaned: обработанные данные
    :param target_column: целевая переменная
    :param id_column: ID Клиента
    :return: предсказанные данные
    """
    # Загрузка обученных моделей
    isolation_forest = joblib.load('./model/isolation_forest_model.pkl')
    elliptic_envelope = joblib.load('./model/elliptic_envelope_model.pkl')

    # Подготовка данных для тестирования (исключаем ID и целевую переменную)
    data_copy = data_cleaned.copy()
    X_test = data_copy.drop(columns=['clnt_id', id_column, 'erly_pnsn_flg', 'age_retirement'])
    y_test = data_copy['erly_pnsn_flg'].apply(lambda x: 1 if x == 0 else -1)  # 1 - класс 0, -1 - класс 1

    # Предсказания каждой модели
    y_pred_if = isolation_forest.predict(X_test)
    y_pred_ee = elliptic_envelope.predict(X_test)

    # Ансамблевое предсказание (голосование)
    predictions = np.vstack((y_pred_if, y_pred_ee)).T
    y_pred = mode(predictions, axis=1)[0].flatten()
    y_pred_final = np.where(y_pred == 1, 0, 1)  # Преобразуем к значениям 0 и 1
    y_test = np.where(y_test == 1, 0, 1)  # Преобразуем к значениям 0 и 1

    # Оценка точности и вывод отчета классификации
    accuracy = accuracy_score(y_test, y_pred_final)
    f1 = f1_score(y_test, y_pred_final, average='weighted')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test, y_pred_final, target_names=['Anomalous', 'Normal']))

    # Создание DataFrame с предсказаниями
    predictions_df = pd.DataFrame({
        'ID клиента': data_cleaned[id_column],
        'Ранний выход на пенсию': y_pred_final,
        'Пенсионный статус': data_cleaned['age_retirement']
    })

    formatted_predictions_df = format_predictions(predictions_df)
    formatted_predictions_df.to_csv('./out/test_predictions_with_ids.csv', index=False, sep=',', encoding='utf-8')

    return formatted_predictions_df


if __name__ == '__main__':
    # data_clean = preprocess_data('./out/cntrbtrs_clnts_ops_trn.csv', './out/trnsctns_ops_trn.csv')
    data_cleaned = preprocess_data('./test_data/cntrbtrs_clnts_ops_tst.csv', './test_data/trnsctns_ops_tst.csv', './test_data/sample_submission.csv')
    test_2(data_cleaned)