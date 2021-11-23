#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import regex as re
import pandas as pd
# import os
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove


from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Здравствуйте! Я ваш виртуальный помощник.')
    reply_keyboard = [['Нужна справка', 'Ипотека'],
                      ['Страхование', 'Перевод/списание/пропажа денег'],
                      ['Уведомления', 'Информация по задолженности'],
                      ['Банковские карты', 'Отсрочка платежа'],
                      ['Кэшбэк бонусы', 'Другой вопрос']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    response = "Пожалуйста, задайте вопрос или выберете тему обращения:"
    update.message.reply_text(response, reply_markup=markup)


def help(update, context):
    """Send a message when the command /help is issued."""
    reply_keyboard = [['Нужна справка', 'Ипотека'],
                      ['Страхование', 'Перевод/списание/пропажа денежных средств'],
                      ['Уведомления', 'Информация по задолженности'],
                      ['Банковские карты', 'Отсрочка платежа'],
                      ['Кэшбэк бонусы', 'Другой вопрос']]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    response = "Задайте вопрос или выберете тему обращения:"
    update.message.reply_text(response, reply_markup=markup)


def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def answer(update, context):
    """find best answer"""

    list_of_greetings = ['Добрый день', 'Добрый вечер', 'Доброй ночи',
                         'День добрый', 'Утро доброе', 'Вечер добрый',
                         'Доброе утро', 'Здравствуйте', 'Привет', 'Добрый ночи',
                         'Подскажите', 'Скажите', 'пожалуйста', 'Доброе Утро', 'Здравствуй',
                         'подскажите', 'скажите', 'Пожалуйста,', 'Добрый ночи', 'здравствуй',
                         'добрый день', 'добрый вечер', 'Скажи', 'Подскажи',
                         'доброй ночи', 'день добрый', 'утро доброе', 'вечер добрый',
                         'доброе утро', 'здравствуйте', 'привет', 'добрый', 'Добрый',
                         'Доброе время суток', 'доброе', 'Доброе',
                         'Доброе время суток', 'Доброго времени суток', 'Всего Вам хорошего',
                         'доброго дня', 'Доброго дня', 'Доброго вечера', 'Здрасти', 'здрасти',
                         'доброго вечера', 'Здраствуйте', 'Здрасьте', 'здраствуйте', 'здрасьте',
                         'Здрасте', 'здрасте', 'Zdravstvuite', 'zdravstvuite', 'Приветствую', 'приветствую',
                         'Пииветсвую']
    list_of_goodbyes = ['Спасибо большое', 'спасибо большое', 'До свидания', 'до свидания',
                        'Спасибо', 'Благодарю', 'До свидание', 'до свидание',
                        'Хорошо', 'Хорошего дня!', 'Всего хорошего Вам',
                        'Всего хорошего', 'Всего доброго', 'Понял', 'Поняла',
                        'Понятно', 'Доброго времени суток', 'Спасибо за помощь',
                        'Спасибо большое', 'Всего хорошего вам', 'Пока', 'пока',
                        'спасибо', 'благодарю', 'спасибо вам большое',
                        'хорошо', 'хорошего дня!', 'всего хорошего',
                        'всего доброго', 'понял', 'поняла', 'Отлично', 'отлично',
                        'понятно', 'доброго времени суток', 'спасибо за помощь',
                        'спасибо большое', 'всего хорошего вам',
                        'всего хорошего Вам', 'Буду ждать', 'буду ждать',
                        'Понятно', 'понятно']
    question_answer_dict = {"как оплатить ежемесячный платеж": "Для списания средств ежемесячного платежа необходимо расположить средства на расчетном счете рублевое РКО. Списание произойдет автоматически",
                            "узнать произошло ли списание по ежемесячному платежу": "Последнее списание средств по ежемесячному платежу было 17.11.2021. Обращаем ваше внимание, что списание платежей происходит в течении двух рабочих дней, после пополнения расчетного счета",
                            "уточнить сумму долга по ипотечному кредиту":  "Ваша сумма долга по ипотечному кредиту составляет 500 000 рублей",
                            "снятие обременения по ипотеке": "Обременение снимается в течении двух рабочих дней после погашения ипотеки. После погашения кредита Вам поступит смс-уведомление или звонок из банка о времени получения Закладной с отметкой Залогодержателя об исполнении обеспеченния",
                            "подключить смс оповещения": "Вам необходимо зайти в настройки в мобильном приложении нашего банка, далее уведомления, далее выбрать 'Уведомления'. Выбрать способ получения 'смс оповещения' ",
                            "отключить смс оповещения": "Вам необходимо зайти в настройки в мобильном приложении нашего банка, далее уведомления, далее выбрать 'Уведомления'. Выбрать 'отключить смс оповещения' ",
                            "не приходят смс оповещения": "Перевожу вас на оператора",
                            "как получить кэшбэк": "Кэшбэк начисляется при совершении покупки кредитной картой нашего банка в магазинах-партнерах. Список таких магазинов можно посмотреть в мобильном приложении нашего банка, в разделе 'Спецпредложения' ",
                            "когда будет начислен кэшбэк": "В течение 30 дней после завершения акции на Вашу карту будет начислен кэшбэк. Если по истечение 30 дней после завершения акции на Вашу карту не будет начислен кэшбэк, просьба обратиться к нам повторно",
                            "на что можно потратить бонусы": "После накопления 500 бонусов их можно потратить на поплнение счета или на дополнительные услуги нашего банка. Более подробную информацию Вы можете найти в мобильном приложении нашего банка в разделе 'Бонусы' ",
                            "узнать срок окончания вклада": "Узнать срок окончания вклада вы можете в мобильном приложении в разделе 'Вклады' ",
                            "закрыть вклад": "Вы можете закрыть банк в любом отделении нашего банка при предъявлении паспорта. Также вы можете закрыть вклад через мобильное приложение, необходимо выбрать Ваш вклад, далее будет опция закрыть.",
                            "справка по платежам и процентам": "Данную справку вы можете получить в приложении в разделе 'Заявления, сообщения'. Далее нажать на карандаш и в разделе 'Получение справок' выбрать 'Справка о произведенных платежах и сумме уплаченных процентов по кредиту' ",
                            "справка об операции": "Данную справку вы можете получить в приложении в разделе 'История'. Нажмите на нужный платёж и следуйте подсказкам",
                            "справка о снятии обременения": "Данную справку вы можете получить в приложении нашего банка в разделе 'Ипотека', Далее нажмите на карандаш в разделе 'Получение справок' и следуйте подсказкам",
                            "правила оформления страхования": "Оформление страхования проводится в офисе банка при обращении с паспортом.",
                            "правила оформления ипотечного кредита": "Оформление ипотечного кредита проводится в офисе банка при обращении с паспортом. При необходимости банк может попросить предоставить дополнительные документы, такие как справка о составе семьи или справка 2-НДФЛ",
                            "стоимость страхования": "Оплата по договору страхования составляет 1411,14 рублей",
                            "дата окончания договора страхования": "Договор страхования жизни заканчивается 12.07.2021. Вам необходимо продлить договор",
                            "cпособы оплаты договора страхования": "Оплатить день в день можно в офисе Банка при обращении с паспортом. Также вы можете произвести оплату в мобильном приложении нашего банка в разделе 'Страхование' ",
                            "аннулирование договора страхования": "Аннулировать текущий договор страхования можно в любом отеделении нашего банка при предъявлении паспорта",
                            "оформить дебетовую карту": "Дебетовую карту можно оформить в любом отделении нашего банка с паспортом либо в нашем мобильном приложении. Для этого зайдите в раздел 'Оформить карту' и следуйте инструкции",
                            "оформить кредитную карту": "Кредитную карту можно оформить в любом отделении нашего банка с паспортом либо в нашем мобильном приложении. Для этого зайдите в раздел 'Оформить карту' и следуйте инструкции",
                            "закрыть карту": "Закрыть карту нашего банка можно в любом отделении нашего банка при предъявлении паспорта",
                            "узнать остаток по счету": "Задолженность по вашему текущему кредиту Вы можете узнать в мобильном приложении нашего банка. Для этого зайдите в раздел 'Расчетный счет' в вашем личном кабинете ",
                            "узнать задолженность по кредиту": "Задолженность по вашему текущему кредиту Вы можете узнать в мобильном приложении нашего банка. Для этого зайдите в раздел 'Кредит' в вашем личном кабинете ",
                            "отсрочка платежа": " Оформить единовременную отсрочку платежа по кредиту можно в мобильном приложении нашего банка в разделе 'Кредиты' ",
                            "реструктуризация кредита": " Реструктуризовать долг можно в любом отделении нашего банка при предъявлении паспорта",
                            "недостаточно средств на счете": "посмотреть историю операций по счету можно в личном кабинете на сайте нашего банка или в мобильном приложении в разделе 'История'",
                            "лишние деньги на счете": "Если вы обнаружили на счету деньги из неизвестного источника, обратитесь в любое отделение нашего банка или позвоните на горячую линию по бесплатному телефону 8 800 0000000 ",
                            "перевод между счетами": " Перевод между счетами можно оформить в мобильном приложении нашего банка в разделе 'Переводы'",
                            "перевод в другой банк": " Перевод в другой банк можно оформить в мобильном приложении нашего банка в разделе 'Переводы'",
                            "сроки для денежных переводов": " Срок перевода зависит от банка-получателя и может составлять от 1 до 5 дней",
                            "потерялась карта": "В случае потери карты, как можно скорее заблокируйте карту в мобильном приложении нашего банка в разделе 'Карты' или позвоните по горячей линии 8 800 0000000",
                            "как дела": "У меня все хорошо, а у тебя?",
                            ", как дела": "У меня все хорошо, а у тебя?",
                            ". как дела": "У меня все хорошо, а у тебя?",
                            "! как дела": "У меня все хорошо, а у тебя?",
                            ", как дела?": "У меня все хорошо, а у тебя?",
                            ". как дела?": "У меня все хорошо, а у тебя?",
                            "! как дела?": "У меня все хорошо, а у тебя?",
                            "другой вопрос": "Перевожу вас на оператора"}

    message = update.message.text

    # Проверяем есть ли приветствие в сообщении. Если есть, то удаляем его, здороваемся и идем дальше
    if any(greeting in message for greeting in list_of_greetings):
        response = "Здравствуйте!"
        update.message.reply_text(response)
        if len(message) <= 5:
            response = "Чем я могу вам помочь?"
            update.message.reply_text(response)
            return
        else:
            message = ''.join([x for x in re.split(r'(\W+)', message) if x not in list_of_greetings])

    # Проверяем есть ли прощание в сообщении. Если есть, прощаемся и выходим из функции
    if any(goodbye in message for goodbye in list_of_goodbyes):
        response = "Спасибо за обращение! Если у Вас возникнут новые вопросы, будем рады Вам помочь!"
        update.message.reply_text(response)
        return

    # Проверяем есть ли ключевые фразы в сообщении. Если есть, то даем ответ по ключевой фразе и выходим
    if any(key in message.lower() for key in question_answer_dict.keys()):
        response = question_answer_dict[message.lower()]
        update.message.reply_text(response)
        return

    # загружаем модель и count_vectorizer
    with open('clf_logreg_vectorizer.pkl', 'rb') as f:
        count_vect, clf_logreg = pickle.load(f)


    df_chat_message = pd.DataFrame()
    df_chat_message['message'] = [message]

    # токенизируем сообщение
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    df_chat_message['message'] = df_chat_message['message'].apply(
        lambda text: ' '.join(tokenizer.tokenize(text.lower())))

    # векторизуем сообщение
    chat_message_count = count_vect.transform(df_chat_message['message'])

    # Классифицируем сообщение
    chat_message_pred = clf_logreg.predict(chat_message_count)

    id2tag = {0: 'certificate', 1: 'mortgage', 2: 'insurance', 3: 'money_transfer', 4: 'notifications',
                  5: 'debt_info', 6: 'bank_card', 7: 'lost_or_extra_money', 8: 'payment_delay', 9: 'cashback'}
    chat_message_tag = [id2tag[id] for id in chat_message_pred]



    intent = chat_message_tag[0]  # model.predict(message)

    if intent == 'mortgage':
        reply_keyboard = [['правила оформления ипотечного кредита', 'как оплатить ежемесячный платеж'],
                          ['уточнить сумму долга по ипотечному кредиту', 'другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        # 'правила оформления ипотечного кредита', 'как оплатить ежемесячный платеж', 'узнать произошло ли списание по ежемесячному платежу'," \
        #            " 'уточнить сумму долга по ипотечному кредиту', 'снятие обременения по ипотеке', 'другой вопрос' "
        update.message.reply_text(response, reply_markup=markup)

    elif intent == 'notifications':
        reply_keyboard = [['Подключить смс оповещения', 'Отключить смс оповещения'],
                          ['Не приходят смс оповещения', 'другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)


    elif intent == 'insurance':
        reply_keyboard = [['правила оформления страхования', 'стоимость страхования'],
                          ['дата окончания договора страхования', 'cпособы оплаты договора страхования'],
                          ['аннулирование договора страхования', 'другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)


    elif intent == 'certificate':
        reply_keyboard = [['справка по платежам и процентам', 'справка об операции'],
                          ['справка о снятии обременения', 'другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)

    elif intent == 'bank_card':
        reply_keyboard = [['оформить дебетовую карту', 'оформить кредитную карту'],
                          ['потерялась карта', 'закрыть карту', 'другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)

    elif intent == 'debt_info':
        reply_keyboard = [['узнать остаток по счету'],
                          ['узнать задолженность по кредиту'],
                          ['другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)

    elif (intent == 'money_transfer') | (intent == 'lost_or_extra_money'):
        reply_keyboard = [['Перевод между счетами','Перевод в другой банк'],
                          ['Сроки для денежных переводов', 'недостаточно средств на счете'],
                          ['лишние деньги на счете', 'другой вопрос']]

        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)

    elif intent == 'cashback':
        reply_keyboard = [['Как получить кэшбэк','Когда будет начислен кэшбэк'],
                          ['На что можно потратить бонусы', 'другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)

    # elif intent == 'lost_or_extra_money':
    #     reply_keyboard = [['недостаточно средств на счете'],
    #                       ['лишние деньги на счете'],
    #                       ['другой вопрос']]
    #     markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    #     response = "Пожулуйста, выберете тему обращения:"
    #     update.message.reply_text(response, reply_markup=markup)

    elif intent == 'payment_delay':
        reply_keyboard = [['отсрочка платежа'],
                          ['реструктуризация кредита'],
                          ['другой вопрос']]
        markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        response = "Пожулуйста, выберете тему обращения:"
        update.message.reply_text(response, reply_markup=markup)

    else:
        response = "К сожалению, я не смог понять тему вашего обращения, перевожу Вас на оператора"
        update.message.reply_text(response)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():

    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("2125929704:AAHYhmEyEsWCCX1R9KPgPiA2oZau3xzVXbc", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, answer))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()