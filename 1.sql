
--Переводим timestamp для таблицы с фактической статистикой в datetime,
--с дальнейшей группировкой значений по часам
With fact_incoming AS (
Select queueId, 
CONVERT(datetime, DATEADD(HOUR, timestamp / 3600000, CAST('1970-01-01 00:00:00' AS datetime))) start_of_hour
, count(new_t) actual
  from dbo.supp_pm_fact_incoming
  group by queueId, CONVERT(datetime, DATEADD(HOUR, timestamp / 3600000, CAST('1970-01-01 00:00:00' AS datetime)))
)

--В качестве оценки рассмотрим Root mean squared error - среднеквадратичная ошибка
--В данном случае, будет более полезна, чем средняя абсолютная ошибка (MAE), 
--поскольку RMSE более чувствителен к сильным отклонениям.
--Расчитаем RMSE для:
--1) Всех значений прогнозов
--2) Для значений прогнозов, которые оказались меньше фактического количества обращений
--3) Для значений прогнозов, которые оказались больше фактического количества обращений
Select queueId, 
sqrt(sum(square_error_1) / count(start_of_hour)) rmse_1,
sqrt(sum(square_error_2) / count(start_of_hour)) rmse_2,
sqrt(sum(underest_error_1) / count(start_of_hour)) underest_rmse_1,
sqrt(sum(underest_error_2) / count(start_of_hour)) underest_rmse_2,
sqrt(sum(overest_error_1) / count(start_of_hour)) overest_rmse_1,
sqrt(sum(overest_error_2) / count(start_of_hour)) overest_rmse_2
from (
Select queueId, start_of_hour,
IIF(prediction1 <= actual, 1 * square(actual - prediction1), 0 * square(actual - prediction1)) underest_error_1,
IIF(prediction2 <= actual, 1 * square(actual - prediction2), 0 * square(actual - prediction2)) underest_error_2,
IIF(prediction1 > actual, 1 * square(actual - prediction1), 0 * square(actual - prediction1)) overest_error_1,
IIF(prediction2 > actual, 1 * square(actual - prediction2), 0 * square(actual - prediction2)) overest_error_2,
square(prediction1 - actual) square_error_1,
square(prediction2 - actual) square_error_2
----Вложенный запрос, объединяющий все три таблицы и удаляющий из дальнейших расчетов часы 
----фактическое/предсказываемая нагрузка по которым отсутствует хотя бы в одной таблице
from (
Select fact.queueId, fact.start_of_hour, fact.actual, pred1.prediction prediction1, pred2.prediction prediction2 
from fact_incoming fact
inner join dbo.supp_pm_prediction_1 pred1 
	on fact.start_of_hour = pred1.start_of_hour 
	and fact.queueId = pred1.queueId
inner join dbo.supp_pm_prediction_2 pred2
	on fact.start_of_hour = pred2.start_of_hour 
	and fact.queueId = pred2.queueId
) as comparison 
) as calculated_errors
group by queueId


--Для 1-ой линии можно говорить о том, что:
--"Прогноз 1" склонен в большей степени недооценивать фактическую нагрузку, по сравнению с "Прогноз 2"
--"Прогноз 2" склонен в большей степени переоценивать фактическую нагрузку, по сравнению с "Прогноз 1"

--Для 10-ой линии наблюдается обратная ситуация:
--"Прогноз 1" склонен в большей степени переоценивать фактическую нагрузку, по сравнению с "Прогноз 2"
--"Прогноз 2" склонен в большей степени недооценивать фактическую нагрузку, по сравнению с "Прогноз 2"

--Если нам выгоднее ситуация, при которой фактическая нагрузка не превышает предсказываемую, чем наоборот,
--то, "Прогноз 1" оказывается более точным для линии №10, 
--а "Прогноз 2" - для линии №1