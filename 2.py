import pandas as pd
from datetime import datetime as dt, timedelta
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#Перевод timestamp в datetime
def parse_timestamp(time_in_seconds):
    return dt.utcfromtimestamp(float(time_in_seconds))

#Для каждого асессора,
# функция определяет наборы пересекающихся или полностью вложенных друг в друга временных интервалов
#Для каждого из таких наборов осуществляется аггрегация входящих в него строк, при которой:
# а) считается количество таких строк,
# б) осуществляется объединение временные интервалов
# в) расчитывается среднее время выполнения задачи, путем деления времени выполнения всех задач из набора на количество задач в наборе
def combine_intervals(df):
    prev_start_ts = df['start_ts'].iloc[0]
    prev_end_ts = df['end_ts'].iloc[0]
    prev_task_count = 0
    prev_login = df['worker'].iloc[0]

    #
    aggregate_dict = {}
    i = 0
    df_last_index = df.last_valid_index()

    # row[1] - login
    # row[3] - task_id
    # row[5] - start_ts
    # row[6] - end_ts

    for row in df.itertuples():
        if row[1] == prev_login and row[5] < prev_end_ts:  # значит, есть пересечение или вложение
            prev_task_count += 1
            prev_end_ts = max(prev_end_ts, row[6])
        else:
            aggregate_dict[i] = {'login': prev_login, 'count': prev_task_count
                , 'start_ts': prev_start_ts, 'end_ts': prev_end_ts}
            i = i + 1
            prev_login, prev_task_count, prev_start_ts, prev_end_ts = row[1], 1, row[5], row[6]
        if row.Index == df_last_index:
            aggregate_dict[i] = {'login': prev_login, 'count': prev_task_count
                ,'start_ts': prev_start_ts, 'end_ts': prev_end_ts}

    aggregated_df = pd.DataFrame.from_dict(aggregate_dict, "index")
    aggregated_df['delta'] = (aggregated_df['end_ts'] - aggregated_df['start_ts']).apply(lambda x: x.total_seconds())
    aggregated_df['time_per_task'] = aggregated_df['delta'] / aggregated_df['count']
    return aggregated_df

#Переведем timestamp во время
df_source = pd.read_csv("./data/pm/task_about_time.tsv", parse_dates=['start_ts', 'end_ts'], delimiter="\t", date_parser=parse_timestamp)

#Проверим, могут ли у одного и того же асессора пересекаться временные промежутки,
#в течение которых он работал одновременно над несколькими заданиями
#Для этого 1) отсортируем значения по ассессору и времени начала работы над заданием
df_source = df_source.sort_values(['worker','start_ts'])

#Определим, существуют ли пересечения временных интервалов для одного асессора пересечение,
#в рамках представленной статистики
#https://stackoverflow.com/questions/42462218/find-date-range-overlap-in-python
#В случае пересечения будут найдутся строки со значением в колонке 'overlap' = True
df_source['overlap'] = (df_source.groupby('worker')
                       .apply(lambda x: (x['end_ts'].shift() - x['start_ts']) > timedelta(0))
                       .reset_index(level=0, drop=True))

print(df_source[df_source['overlap'] == True]['task_id'].any())
# df.to_csv('../data/pm/task_about_time_results.csv')

#Такие пересечения имеются.
#Рассмотрим статистику по каждому из типов заданий по отдельности.
#Построим гистрограммы распределения, оценим ее форму.

df_source_1 = df_source[df_source['project_id'] == 1]
df_source_2 = df_source[df_source['project_id'] == 2]

aggregate_1 = combine_intervals(df_source_1)
aggregate_2 = combine_intervals(df_source_2)
sample_1 = aggregate_1['time_per_task']
sample_2 = aggregate_2['time_per_task']

#Построим гистрограмму распределения количества асессоров,
#выполнивших задание за время, входящее в определенный диапазон
#Число интервалов (корзин) определим в зависимости от числа N
# #https://ami.nstu.ru/~headrd/seminar/xi_square/28.htm
# sample = sample_1
# #sample = sample_2
# N = len(sample)
# bin_amount = np.round(1.72 * np.power(N,1/3),0).astype(int)
# ax = sample.hist(bins=bin_amount*3)
# ax.set_xlim([0,max(sample)])
# plt.show(ax=ax)


# По построенным гистограммам можно сделать вывод об
# 1) явной ассиметрии
# 2) о наличии выбросов с большей стороны
# Данные вывод справедливы для обеих гистограмм, т.е. для каждого из типа заданий

# В качестве "справедливой" рассмотрим среднюю оценку времени,
# потраченную асессорами на выполнение заданий определенного типа,
# умноженную на N/30 (согласно условию задания)

# При этом, в качестве числовой характеристики, описывающей "среднее" время выполнения задания определенного типа
# имеет смысл рассмотреть медиану, как характеристику, более устойчивую к наличию выбросов

print('median1:' + str(sample_1.median()))
print('median2:' + str(sample_2.median()))
print('mean1:' + str(sample_1.mean()))
print('mean2:' + str(sample_2.mean()))
print()
# Помимо выбросов, можно обратить внимание на завышение времени выполнения части
# заданий, т.к. возможно пересечение временных интервалов при выполнении задач разных типов одним асессором.
# В качестве способа отсечения выбросов, рассмотрим характеристики усеченного
# вариационного ряда, а именно значений, не превышающих значение 0.75 квантиля
# (т.е возьмем 75% самых лучших результатов)

sample_1_cut_q75 = sample_1[sample_1 < sample_1.quantile(0.75)]
sample_2_cut_q75 = sample_2[sample_2 < sample_2.quantile(0.75)]

#Построим гистограммы для каждого из усеченных рядов
plot_sample = sample_2_cut_q75
N = len(plot_sample)
bin_amount = np.round(1.72 * np.power(N,1/3),0).astype(int)
ax = plot_sample.hist(bins=bin_amount)
ax.set_xlim([0,max(plot_sample)])
plt.show(ax=ax)


#Т.к. мы расчитываем значения для каждого из типа заданий отдельно, и условный "вес" каждой задачи
#внутри рассматриваемой категории постоянен, мы можем рассмотреть медиану в качестве "справедливой"
# средней оценки относительно всех асессоров

#Пересчитаем характеристики
print('sample_1_q75_median:' + str(sample_1_cut_q75.median()))
print('sample_2_q75_median:' + str(sample_2_cut_q75.median()))
print('sample_1_q75_mean:' + str(sample_1_cut_q75.mean()))
print('sample_2_q75_mean:' + str(sample_2_cut_q75.mean()))
print()

#В качестве справедливой оплаты возьмем
print(str(sample_1_cut_q75.median() / 30) + 'N')
print(str(sample_2_cut_q75.median() / 30) + 'N')