import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy.stats import mode
from preproccesing import preprocess_data

def train_and_save_models(data_cleaned, target_column='erly_pnsn_flg', id_column='accnt_id'):
    """
    Данная функция обучает модель и сохраняет ее для дальнейшего использования
    :param data_cleaned: обработанные данные
    :param target_column: целевая переменная
    :param id_column: ID счета клиента
    :return:
    """
    # Создаем копию данных для обучения
    data_copy = data_cleaned.copy()
    data_copy.drop(columns=['age_retirement'], inplace=True)

    # Выделяем нормальный класс для обучения (предполагается, что target=0 это нормальные данные)
    normal_class_data = data_copy[data_copy[target_column] == 0].drop(columns=[id_column, 'clnt_id', target_column])

    # Определение моделей для ансамблирования
    isolation_forest = IsolationForest(contamination=0.01, random_state=42)
    elliptic_envelope = EllipticEnvelope(contamination=0.01, random_state=42)

    # Обучение моделей на нормальных данных
    isolation_forest.fit(normal_class_data)
    elliptic_envelope.fit(normal_class_data)

    # Сохранение моделей в формате .pkl
    joblib.dump(isolation_forest, './model/isolation_forest_model.pkl')
    joblib.dump(elliptic_envelope, './model/elliptic_envelope_model.pkl')

    # Подготовка полного набора данных для предсказания (без ID и целевой переменной)
    X_all = data_copy.drop(columns=['clnt_id', id_column, target_column, 'age_retirement'])
    y_all = data_copy[target_column].apply(lambda x: 1 if x == 0 else -1)  # 1 - нормальные, -1 - аномалии

    # Предсказания каждой модели
    y_pred_if = isolation_forest.predict(X_all)
    y_pred_ee = elliptic_envelope.predict(X_all)

    # Ансамблевое предсказание (голосование)
    predictions = np.vstack((y_pred_if, y_pred_ee)).T
    y_pred = mode(predictions, axis=1)[0].flatten()
    y_pred_final = np.where(y_pred == 1, 0, 1)  # Преобразуем к значениям 0 и 1

    # Создание DataFrame с предсказаниями
    predictions_df = pd.DataFrame({
        id_column: data_cleaned[id_column],
        target_column: y_pred_final
    })

    # Оценка точности и вывод отчета классификации
    accuracy = accuracy_score(y_all, y_pred)
    f1 = f1_score(y_all, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_all, y_pred, target_names=['Anomalous', 'Normal']))

    # Сохранение предсказаний в CSV
    predictions_df.to_csv('predictions_with_ids.csv', index=False, sep=',', encoding='utf-8')

    return predictions_df

if __name__ == '__main__':
    data_cleaned = preprocess_data('./train_data/cntrbtrs_clnts_ops_trn.csv', './train_data/trnsctns_ops_trn.csv', './train_data/sample_submission.csv')
    train_and_save_models(data_cleaned)