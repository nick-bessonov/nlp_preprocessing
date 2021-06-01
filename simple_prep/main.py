# Разработчик ноды: Бессонов Никита
# Нода приводит текст к нижнему регистру и удаляет все ненужные символы, цифры (опционально)
# Все преобразования сохраняются в дополнительный столбец (нужно указать)

import re
import pandas as pd


# Приведение строк к нижнему регистру
def lower_case(data, text_field, new_field):
    data[new_field] = data[text_field].str.lower()

    return data


# замена подряд идущих пробелов, знаков табуляции на один пробельный символ
def del_spaces(data, text_field):
    data[text_field].replace(to_replace='\s\s+', value=' ', regex=True, inplace=True)
    return data


# удаление разных символов не являющиеся цифрами и буквами
def trash_chars(data,
                text_field,
                need_del_dash=True,  # нужно ли удалять тире
                need_del_number=False,  # нужно ли удалять все цифры
                need_del_in_brackets=True,  # нужно ли удалять все внутри скобок
                need_del_eng=True,  # нужно ли удалять английские буквы
                ):
    num_reg = r'\d+'  # регулярка для чисел
    month_reg = r'январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь'
    month_reg = re.compile(month_reg, re.IGNORECASE)

    # удаление всего что стоит в скобках
    if need_del_in_brackets:
        data[text_field].replace(to_replace='\([^\(\)]+\)', value=' ', regex=True, inplace=True)
    else: pass

    # замена английской с на русскую и ё на е
    data[text_field].replace({'c': 'с', 'C': 'С'}, regex=True, inplace=True)
    data[text_field].replace({'ё': 'е', 'Ё': 'Е'}, regex=True, inplace=True)

    # удаление всех символов, кроме букв, цифр, пробелов, тире
    if need_del_eng:
        data[text_field].replace(to_replace='[^А-Яа-яЁё\s\d-]', value=' ', regex=True, inplace=True)
    else:
        data[text_field].replace(to_replace='[^А-Яа-яЁёA-Za-z\s\d-]', value=' ', regex=True, inplace=True)

    if need_del_number:
        data[text_field] = data[text_field].apply(lambda x: re.sub(num_reg, ' ', x))
    else: pass

    if need_del_dash:
        data[text_field].replace(to_replace='-', value='', regex=True, inplace=True)
    else: pass

    data = del_spaces(data, text_field)

    # удаление пустых ячеек после преобразований
    data = data[(data[text_field].notna()) | (data[text_field] != '')]

    return data


# функция консолидации и вывода результата
def main(train,
         oos=None,
         oot=None,
         text_field=None,
         new_field=None,
         need_del_dash=True,
         need_del_number=False,
         need_del_in_brackets=True,
         need_lower_case=True,  # Нужно ли приводить к нижнему регистру
         need_del_eng=True
         ):
    results = dict()

    # создаем новый столбец
    train[new_field] = train[text_field]
    test_empty = pd.DataFrame(columns=train.columns)

    # проверяем на наличие тестовых выборок
    if oos is not None:
        oos[new_field] = oos[text_field]
    else:
        oos = test_empty

    if oot is not None:
        oot[new_field] = oot[text_field]
    else:
        oot = test_empty

    if need_lower_case:
        train = lower_case(train, text_field, new_field)
        oos = lower_case(oos, text_field, new_field)
        oot = lower_case(oot, text_field, new_field)
    else:
        pass

    train = trash_chars(train, new_field, need_del_dash, need_del_number, need_del_in_brackets, need_del_eng)
    oos = trash_chars(oos, new_field, need_del_dash, need_del_number, need_del_in_brackets, need_del_eng)
    oot = trash_chars(oot, new_field, need_del_dash, need_del_number, need_del_in_brackets, need_del_eng)

    results['output_train'] = train
    results['output_oos'] = oos
    results['output_oot'] = oot
    results['output_table_head'] = {'table_data': train.head().to_dict('records')}

    return results
