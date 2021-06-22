import re
import pandas as pd
import pymorphy2
import stop_words
from spellchecker import SpellChecker  # библиотека используется для удаления слов с ошибками
from pyaspeller import YandexSpeller  # библиотека используется для замены слов с ошибками (работает в разы быстрее спеллчекера)
from functools import lru_cache


# Приведение строк к нижнему регистру
def lower_case(data, text_field):
    data[text_field] = data[text_field].str.lower()
    return data


# замена подряд идущих пробелов, знаков табуляции на один пробельный символ
def del_spaces(data, text_field):
    data[text_field].replace(to_replace='\s\s+', value=' ', regex=True, inplace=True)
    return data


# удаление разных символов не являющиеся цифрами и буквами
def trash_chars(data,
                text_field,
                need_lower_case=True,  # приведение к нижнему регистру
                need_del_dash=True,  # нужно ли удалять тире
                need_del_number=False,  # нужно ли удалять все цифры
                need_del_in_brackets=True,  # нужно ли удалять все внутри скобок
                need_del_eng=True,  # нужно ли удалять английские буквы
                ):
    num_reg = r'\d+'  # регулярка для чисел
    # month_reg = r'январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь'
    # month_reg = re.compile(month_reg, re.IGNORECASE)

    if need_lower_case:
        data = lower_case(data, text_field)

    # удаление всего что стоит в скобках
    if need_del_in_brackets:
        data[text_field].replace(to_replace='\([^\(\)]+\)', value=' ', regex=True, inplace=True)
    else:
        pass

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
    else:
        pass

    if need_del_dash:
        data[text_field].replace(to_replace='-', value='', regex=True, inplace=True)
    else:
        pass

    data = del_spaces(data, text_field)

    # удаление пустых ячеек после преобразований
    data = data[(data[text_field].notna()) | (data[text_field] != '')]

    return data


# Удаление стоп-слов
def remove_stop_words(data, text_field, mode, new_stopwords):
    # подгружаем стоп-слова из файла
    stopwords = stop_words.stopwords

    if mode is not None:
        if mode:
            stopwords = stopwords + new_stopwords
        else:
            stopwords = new_stopwords

    # print('List: {}'.format(stopwords))

    stop_words_to_reg = r'\b|\b'.join(stopwords)  # переводим список стопслов в регулярку
    stop_words_to_reg = r'\b' + stop_words_to_reg + r'\b'  # для граничных символов

    data[text_field] = data[text_field].apply(lambda x: re.sub(stop_words_to_reg, '', x))

    return data


# кеширует результат тяжелой функици, в теории должно ускорить лемматизацию
@lru_cache(maxsize=100000)
def parse_lru_cache(text, pymorph):
    return pymorph.parse(text)[0]


# вариант без кеширования
def parse_without_cache(text, pymorph):
    return pymorph.parse(text)[0]


# нахождение нормальной формы слова и замена ё на е
def lemmatization_token(token):
    new_token = token.normal_form.replace('ё', 'е')

    return new_token


# Поиск опечаток в слове, удаление/замена
def spell_check_token(token, russian, need_del):
    correction_form = russian.correction(token)

    if correction_form != token:
        token = '' if need_del else correction_form
    else:
        pass

    return token


# Принадлежность слова к одной из сущности, замена / удаление
def ner_token(token,
              parsed_token,
              need_del_name,
              need_del_geo,
              need_del_org,
              need_del_numb):
    replaces = {
        'Surn': ['NAME', need_del_name],
        'Name': ['NAME', need_del_name],
        'Patr': ['NAME', need_del_name],  # отчество
        'Geox': ['GEO', need_del_geo],
        'NUMR': ['NUMBER', need_del_numb],
        'NUMB': ['NUMBER', need_del_numb],
        'Orgn': ['ORGANIZATION', need_del_org]
    }

    new_token = token

    for tag, item in replaces.items():
        if tag in parsed_token.tag:
            if item[1] is not None:
                new_token = '' if item[1] else item[0]  # (True) удаление сущности | (False) замена
                break
            else:
                break
        else:
            pass

    return new_token


# удаление подряд идущих сущностей
def del_repeat_entity(text):
    entity_tokens = ['NAME', 'GEO', 'NUMBER', 'ORGANIZATION']

    for ent in entity_tokens:
        reg_ex = '(%s\s){2,}|(\s%s){2,}' % (ent, ent)  # регулярка для подряд идущих сущностей
        text = re.sub(reg_ex, ' ' + ent + ' ', text)

    text = re.sub('\s\s+', ' ', text)  # удаляем лишние пробелы

    return text


# основная функция, в которой происходит преобразования всего датафрейма по указанному столбцу
def lemmatizer_ner_spellchecker(train,
                                oos,
                                oot,
                                text_field,
                                need_spellchecker,
                                need_lemma,
                                need_ner,
                                need_del_name,
                                need_del_geo,
                                need_del_org,
                                need_del_numb,
                                need_del_spell,
                                need_lru_cache,
                                need_del_months):
    # преобразование каждой строки
    def string_preprocessing(text,
                             need_spellchecker,
                             need_lemma,
                             need_ner,
                             need_del_name,
                             need_del_geo,
                             need_del_org,
                             need_del_numb,
                             need_del_spell,
                             need_lru_cache,
                             pymorph,
                             russian,
                             speller):

        # проверка на ошибки с ипользованием яндекс чекера / замена на правильно написанные слова
        if need_spellchecker and not need_del_spell:
            text = speller.spelled(text)  # работает быстрее, чем проверять по каждому слову в отдельности

        # разбиваем текст на токены в виде списка
        tokens = [token for token in text.lower().split(' ') if token not in ['']]

        # список куда будут сохраняться измененные токены
        pm_tokens = []

        for token in tokens:

            # при удалении слов с ошибками используется другой алгоритм
            if need_spellchecker and need_del_spell:
                token = spell_check_token(token, russian, need_del_spell)  # работает в разы медленне, чем от яндекса
            else:
                pass

            # потом находим его нормальную форму
            if need_lemma:
                parsed_token = parse_lru_cache(token, pymorph) if need_lru_cache else parse_without_cache(token,
                                                                                                          pymorph)
                token = lemmatization_token(parsed_token)
            else:
                pass

            # и только потом ищем сущности
            if need_ner:
                token = ner_token(token,
                                  parsed_token,
                                  need_del_name,
                                  need_del_geo,
                                  need_del_org,
                                  need_del_numb)
            else:
                pass

            # сохраняем все изменяния в список для последующей склейки
            pm_tokens += [token]

        new_text = re.sub('\s\s+', ' ', ' '.join(pm_tokens))

        # удаление подряд идущих сущностей
        if need_ner:
            new_text = del_repeat_entity(new_text)

        # возвращаем все в одну строку, заодно подчищаем лишние пробелы
        return new_text

    pymorph = pymorphy2.MorphAnalyzer()
    russian = SpellChecker(language='ru')
    speller = YandexSpeller()

    # обращаемся к каждой ячейке
    train[text_field] = train[text_field].apply(lambda x: string_preprocessing(str(x),
                                                                               need_spellchecker,
                                                                               need_lemma,
                                                                               need_ner,
                                                                               need_del_name,
                                                                               need_del_geo,
                                                                               need_del_org,
                                                                               need_del_numb,
                                                                               need_del_spell,
                                                                               need_lru_cache,
                                                                               pymorph,
                                                                               russian,
                                                                               speller))

    if need_del_months:  # удаление месяцев
        # регулярка для удаления
        month_reg = r'январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь'
        month_reg = re.compile(month_reg, re.IGNORECASE)
        train[text_field] = train[text_field].apply(lambda x: re.sub(month_reg, ' ', x))

    # удаление пустых ячеек после преобразований
    train = train[(train[text_field].notna()) | (train[text_field] != '')]

    # делаем все тоже самое только для тестовых выборок
    if len(oos.index) != 0:
        oos[text_field] = oos[text_field].apply(lambda x: string_preprocessing(str(x),
                                                                               need_spellchecker,
                                                                               need_lemma,
                                                                               need_ner,
                                                                               need_del_name,
                                                                               need_del_geo,
                                                                               need_del_org,
                                                                               need_del_numb,
                                                                               need_del_spell,
                                                                               need_lru_cache,
                                                                               pymorph,
                                                                               russian))
        if need_del_months:  # удаление месяцев
            month_reg = r'январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь'
            month_reg = re.compile(month_reg, re.IGNORECASE)
            oos[text_field] = oos[text_field].apply(lambda x: re.sub(month_reg, ' ', x))

        oos = oos[(oos[text_field].notna()) | (oos[text_field] != '')]

    if len(oot.index) != 0:
        oot[text_field] = oot[text_field].apply(lambda x: string_preprocessing(str(x),
                                                                               need_spellchecker,
                                                                               need_lemma,
                                                                               need_ner,
                                                                               need_del_name,
                                                                               need_del_geo,
                                                                               need_del_org,
                                                                               need_del_numb,
                                                                               need_del_spell,
                                                                               need_lru_cache,
                                                                               pymorph,
                                                                               russian))
        if need_del_months:  # удаление месяцев
            month_reg = r'январь|февраль|март|апрель|май|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь'
            month_reg = re.compile(month_reg, re.IGNORECASE)
            oot[text_field] = oot[text_field].apply(lambda x: re.sub(month_reg, ' ', x))

        oot = oot[(oot[text_field].notna()) | (oot[text_field] != '')]

    return train, oos, oot


# функция консолидации и вывода результата
def nlp_preprocessing(
        # Основные параметры
        train,
        oos=None,
        oot=None,
        text_field=None,
        target_name=None,  # ?? (потом удалить)

        # Простой препроцессинг
        need_del_dash=True,  # удаление тире
        need_del_number=False,
        need_del_in_brackets=True,  # удаление внутри скобок
        need_lower_case=True,  # Нужно ли приводить к нижнему регистру
        need_del_eng=True,

        # Спеллчекер (поиск опечаток)
        need_spellchecker=None,
        need_del_spell='nothing',

        # Лемматизация
        need_lemma=None,
        need_lru_cache=False,

        # Поиск сущностей (NER)
        need_ner=None,
        need_del_number_ner='nothing',
        need_del_name='nothing',
        need_del_org='nothing',
        need_del_geo='nothing',
        need_del_months=False,

        # Удаление стопслов
        need_del_stopwords=None,
        new_stopwords=None,
        mode_stopwords='delete default list'
):
    new_text_field = 'new_prep_' + text_field

    # действия: удаление / замена / ничего
    del_or_replace = {'delete': True,
                      'replace': False,
                      'nothing': None
                      }

    option_stopwords = {'default list': None,
                        'default list + additional list': True,
                        'only additional list': False
                        }

    need_del_spell = del_or_replace[need_del_spell]
    need_del_number_ner = del_or_replace[need_del_number_ner]
    need_del_name = del_or_replace[need_del_name]
    need_del_org = del_or_replace[need_del_org]
    need_del_geo = del_or_replace[need_del_geo]
    mode_stopwords = option_stopwords[mode_stopwords]

    if not need_lemma and need_ner:
        raise ValueError('Please choose lemmatization, before using NER!')

    # создаем новый столбец
    train[new_text_field] = train[text_field]
    test_empty = pd.DataFrame(columns=train.columns)  # ??

    # проверяем на наличие тестовых выборок
    if oos is not None:
        oos[new_text_field] = oos[text_field]
    else:
        oos = test_empty

    if oot is not None:
        oot[new_text_field] = oot[text_field]
    else:
        oot = test_empty

    # Простой препроцессинг
    train = trash_chars(train, new_text_field, need_lower_case, need_del_dash, need_del_number, need_del_in_brackets,
                        need_del_eng)
    oos = trash_chars(oos, new_text_field, need_lower_case, need_del_dash, need_del_number, need_del_in_brackets,
                      need_del_eng)
    oot = trash_chars(oot, new_text_field, need_lower_case, need_del_dash, need_del_number, need_del_in_brackets,
                      need_del_eng)

    # Поиск опечаток, лемматизация, сущности
    if need_spellchecker or need_lemma or need_ner:
        train, oos, oot = lemmatizer_ner_spellchecker(train,
                                                      oos,
                                                      oot,
                                                      new_text_field,
                                                      need_spellchecker,
                                                      need_lemma,
                                                      need_ner,
                                                      need_del_name,
                                                      need_del_geo,
                                                      need_del_org,
                                                      need_del_number_ner,
                                                      need_del_spell,
                                                      need_lru_cache,
                                                      need_del_months)
    else:
        pass

    #  Удаление стопслов
    if need_del_stopwords:
        # stop_words = open('stop_words.txt', 'r').read().splitlines()  # дополненный список стопслов из nltk
        train = remove_stop_words(train, new_text_field, mode_stopwords, new_stopwords)
        oos = remove_stop_words(oos, new_text_field, mode_stopwords, new_stopwords)
        oot = remove_stop_words(oot, new_text_field, mode_stopwords, new_stopwords)
    else:
        pass

    return train, oos, oot
