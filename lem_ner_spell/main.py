# Разработчик ноды: Бессонов Никита
# Нода 1) удаляет стоп-слова 2) удаляет/заменяет именованные сущности
# 3) Заменяет/удаляет слова с опечаткой
# Все преобразования сохраняются в допольнительный столбец (нужно указать)

import re
import pandas as pd
import ast
import pymorphy2
import stop_words
from spellchecker import SpellChecker
from functools import lru_cache


# Удаление стоп-слов
def remove_stop_words(data, text_field, mode, new_stopwords):
    # подгружаем стоп-слова из файла
    stopwords = stop_words.stopwords

    if mode is not None:
        if mode:
            stopwords = stopwords + new_stopwords
        else:
            stopwords = new_stopwords

    print('List: {}'.format(stopwords))

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
                             russian):

        # разбиваем текст на токены в виде списка
        tokens = [token for token in text.lower().split(' ') if token not in ['']]

        # список куда будут сохраняться измененные токены
        pm_tokens = []

        for token in tokens:

            # сначала исправляем слово с ошибками
            if need_spellchecker:
                token = spell_check_token(token, russian, need_del_spell)
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
                                                                               russian))

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
def main(train,
         oos=None,
         oot=None,
         text_field=None,
         new_field=None,
         need_spellchecker=None,
         need_lemma=None,
         need_ner=None,
         need_del_number=None,
         need_del_name=None,
         need_del_org=None,
         need_del_geo=None,
         need_del_stopwords=None,
         need_del_spell=None,
         need_lru_cache=False,
         need_del_months=False,
         new_stopwords=None,
         mode_stopwords=None):
    del_or_replace = {'delete': True,
                      'replace': False,
                      'nothing': None
                      }

    option_stopwords = {'delete default list': None,
                        'delete default list + additional list': True,
                        'delete only additional list': False
                        }

    need_del_number = del_or_replace[need_del_number]
    need_del_name = del_or_replace[need_del_name]
    need_del_org = del_or_replace[need_del_org]
    need_del_geo = del_or_replace[need_del_geo]
    need_del_spell = del_or_replace[need_del_spell]
    mode_stopwords = option_stopwords[mode_stopwords]

    new_stopwords = ast.literal_eval(new_stopwords)  # преобразование в список

    if not need_lemma and need_ner:
        raise ValueError('Please choose lemmatization, before using NER!')

    # создаем новый столбец
    train[new_field] = train[text_field].copy()
    test_empty = pd.DataFrame(columns=train.columns)

    # проверка на наличие тестовых выборок
    if oos is not None:
        oos[new_field] = oos[text_field]
    else:
        oos = test_empty

    if oot is not None:
        oot[new_field] = oot[text_field]
    else:
        oot = test_empty

    if need_spellchecker or need_lemma or need_ner:
        train, oos, oot = lemmatizer_ner_spellchecker(train,
                                                      oos,
                                                      oot,
                                                      new_field,
                                                      need_spellchecker,
                                                      need_lemma,
                                                      need_ner,
                                                      need_del_name,
                                                      need_del_geo,
                                                      need_del_org,
                                                      need_del_number,
                                                      need_del_spell,
                                                      need_lru_cache,
                                                      need_del_months)
    else:
        pass

    if need_del_stopwords:
        # stop_words = open('stop_words.txt', 'r').read().splitlines()  # дополненный список стопслов из nltk
        train = remove_stop_words(train, new_field, mode_stopwords, new_stopwords)
        oos = remove_stop_words(oos, new_field, mode_stopwords, new_stopwords)
        oot = remove_stop_words(oot, new_field, mode_stopwords, new_stopwords)
    else:
        pass

    results = dict()
    results['output_train'] = train
    results['output_oos'] = oos
    results['output_oot'] = oot
    results['output_table_head'] = {'table_data': train.head().to_dict('records')}

    return results
