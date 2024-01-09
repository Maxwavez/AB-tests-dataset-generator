from flask import Flask, render_template, send_file
import hashlib
import uuid
import numpy as np
import random
import pandas as pd
from scipy.stats import gamma
import zipfile
import os
import tempfile


app = Flask(__name__)

def get_parameters(n=50000):
    '''
    Функция возвращает параметры для генерации распределения:
    basic_conv - вероятность конверсии
    probability - вероятность изменения конверсии в тестовой выборке
    '''
    # генерируем размер выборки
    n = n

    # задаем базовую конверсию
    basic_cr = round(random.uniform(0.2, 0.6), 2)

    # генерируем факт изменения конверсии (0 - значит изменений не будет, 1 - конверсия изменится)
    if random.randint(0, 1) == 1:
        # генерируем случайное изменение конверсии в заданном диапазоне
        new_cr = basic_cr + round(random.uniform(0.02, 0.1), 2)
    else:
        new_cr = basic_cr

    n, control_cr, test_cr = n, basic_cr, new_cr
    return n, control_cr, test_cr

def get_dataframe(n, control_cr, test_cr):
    '''
    Функция принимает параметры: размер выборки, значение конверсии в контрольной и тестовой группе и генерирует датафрейм с данными о конверсии пользователей (для контроля и теста может отличаться в зависимости
    от параметров)
    '''
    # списки с id
    ids = []

    # Шаг 1. генерация id
    for _ in range(n):
        ids.append(uuid.uuid4().hex)


    # Шаг 2. Хэширование и распределение по группам
    treatment_group = []
    control_group = []

    # задаем соль для сплитования
    salt = uuid.uuid4().hex

    # хэшируем id и распределяем в группы
    for i, id in enumerate(ids):
        if int(hashlib.md5((id + salt).encode()).hexdigest(), 16) % 2 == 0:
            treatment_group.append(id)
        else:
            control_group.append(id)


    # Шаг 3. Присвоение флагов конверсии.

    # для контрольной группы
    control_flag = []
    for i in range(len(control_group)):
        if random.random() <= control_cr:
            # Set the flag to '1' (or any other desired value)
            control_flag.append(1)
        else:
            control_flag.append(0)

    # для тестовой группы
    # если значение конверсии в тестовой группе согласно параметрам должно быть иным (test_cr > control_cr), необходимо изменить значения флагов
    # в ином случаем оставляем значение как в контроле

    test_flag = []
    for i in range(len(treatment_group)):
        if random.random() <= test_cr:
            # Set the flag to '1' (or any other desired value)
            test_flag.append(1)
        else:
            test_flag.append(0)

    # объединяем id и флаги конверсии в датафрейм
    control_users_flag = pd.DataFrame({'id': control_group,
                                 'flag':control_flag})


    treatment_users_flag = pd.DataFrame({'id': treatment_group,
                                 'flag':test_flag})


    # объединяем датафреймы
    users = pd.concat([control_users_flag, treatment_users_flag]).sort_values('id').reset_index(drop=True)


    # Шаг 4. создаем отдельный датафрейм для определения групп пользователей
    control_users = pd.DataFrame({'id': control_group,
                                 'group':'control'})

    test_users = pd.DataFrame({'id': treatment_group,
                                 'group':'treatment'})

    groups = pd.concat([control_users, test_users])


    # Шаг 5. присваиваем значения сумм транзакций для пользователей с конверсией 1
    # Параметры гамма-распределения
    shape = 2 # Параметр формы (можно настраивать)
    scale = 2000  # Параметр масштаба (увеличивайте для больших значений)

    # Генерируем выборку с гамма-распределением
    gamma_data = gamma.rvs(shape, loc=0, scale=scale, size=len(users[users['flag'] == 1]))

    # во временный датафрейм сложим id пользователей, для которых необходимо сгенерировать суммы транзакций и добавим этим суммы
    tmp_df = users[users['flag'] == 1][['id']]
    tmp_df['amount'] = gamma_data
    tmp_df['amount'] = round(tmp_df['amount'])

    # добавляем данные с суммами в общий датафрейм
    transactions = users.merge(tmp_df, on='id', how='left')


    # Шаг 6. Изменение сумм транзакция для тестовой группы (факт различия определить как 50 на 50, изменить на константу 300)
    if random.randint(0, 1) == 1:
        transactions.loc[transactions['id'].isin(groups[groups['group'] == 'treatment']['id']) & transactions['flag'] == 1, 'amount'] = (
            transactions.loc[transactions['id'].isin(groups[groups['group'] == 'treatment']['id']) & transactions['flag'] == 1, 'amount'] + 300
        )


    # возвращаем 2 датафрейма (данные с транзакциями и данные с принадлежностью к группе)
    return transactions[['id', 'amount']], groups


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_dataframe')
def get_dataframe_route():
    n, control_cr, test_cr = get_parameters()
    dataset1, dataset2 = get_dataframe(n, control_cr, test_cr)

    dataset1.to_csv('generated_dataset1.csv', index=False)
    dataset2.to_csv('generated_dataset2.csv', index=False)


    # Создаем временный zip-архив
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip_file:
        with zipfile.ZipFile(temp_zip_file, 'w') as zipf:
            # Записываем первый датасет в CSV внутри zip-файла
            csv_bytes1 = dataset1.to_csv(index=False).encode('utf-8')
            zipf.writestr('generated_dataset1.csv', csv_bytes1)

            # Записываем второй датасет в CSV внутри zip-файла
            csv_bytes2 = dataset2.to_csv(index=False).encode('utf-8')
            zipf.writestr('generated_dataset2.csv', csv_bytes2)

            # Удаляем CSV-файлы, если они больше не нужны
            os.remove('generated_dataset1.csv')
            os.remove('generated_dataset2.csv')

    # Отправляем zip-файл пользователю и удаляем его
    return send_file(temp_zip_file.name, as_attachment=True, download_name='generated_datasets.zip')


if __name__ == '__main__':
    app.run(debug=True)