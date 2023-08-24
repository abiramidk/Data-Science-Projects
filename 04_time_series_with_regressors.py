# Databricks notebook source
# MAGIC %md ##Forecasting using Time Series Analysis with Weather Regressors
# MAGIC 
# MAGIC In this notebook, we will build on the work of the previous one, incorporating hourly temperature and precipitation measurements as regressors into our time series model.  As before, we will start by installing the libraries needed for this work: 

# COMMAND ----------

# load fbprophet library
dbutils.library.installPyPI('FBProphet', version='0.5') # find latest version of fbprophet here: https://pypi.org/project/fbprophet/
dbutils.library.installPyPI('holidays', version='0.9.12') # this line is in response to this issue with fbprophet 0.5: https://github.com/facebook/prophet/issues/1293
dbutils.library.installPyPI('mlflow')
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md And now we will define our function.  Please note that in order to generate a prediction, future (predicted) values for temperature and precipitation must be provided.  The DataFrame expected by this function includes records for both historical and future periods, with the former identified by a value of 1 in the *is_historical* field:

# COMMAND ----------

import mlflow
import mlflow.sklearn
import shutil

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

import pandas as pd

import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

from fbprophet import Prophet

# structure of the dataset returned by the function
result_schema =StructType([
  StructField('station_id',IntegerType()),
  StructField('ds',TimestampType()),
  StructField('y', FloatType()),
  StructField('yhat', FloatType()),
  StructField('yhat_lower', FloatType()),
  StructField('yhat_upper', FloatType()),
  StructField('trend',FloatType()),
  StructField('trend_lower', FloatType()),
  StructField('trend_upper', FloatType()),
  StructField('multiplicative_terms', FloatType()),
  StructField('multiplicative_terms_lower', FloatType()),
  StructField('multiplicative_terms_upper', FloatType()),
  StructField('daily', FloatType()),
  StructField('daily_lower', FloatType()),
  StructField('daily_upper', FloatType()),
  StructField('weekly', FloatType()),
  StructField('weekly_lower', FloatType()),
  StructField('weekly_upper', FloatType()),
  StructField('yearly', FloatType()),
  StructField('yearly_lower', FloatType()),
  StructField('yearly_upper', FloatType()),
  StructField('additive_terms', FloatType()),
  StructField('additive_terms_lower', FloatType()),
  StructField('additive_terms_upper', FloatType()),
  StructField('holidays', FloatType()),
  StructField('holidays_lower', FloatType()), 
  StructField('holidays_upper', FloatType())
  ])

# forecast function
@pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
def get_forecast(keys, group_pd):
  
  # DATA PREP
  # ---------------------------------
  # identify station id and hours to forecast
  station_id = keys[0]
  hours_to_forecast=keys[1]
  
  # extract valid historical data
  history_pd = group_pd[group_pd['is_historical']==1].dropna()
  
  # acquire holidays
  holidays_pd=holidays_broadcast.value
  # ---------------------------------  
  
  # TRAIN MODEL
  # ---------------------------------  
  # configure model
  model = Prophet(
    interval_width=0.80,
    growth='linear',
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    holidays=holidays_pd
    )
  
  # identify the weather regressors
  model.add_regressor('temp_f', mode='multiplicative')
  model.add_regressor('precip_in', mode='multiplicative')

  # train model
  model.fit( history_pd )

  # save models for potential later use
  model_path = '/dbfs/mnt/citibike/timeseries_regressors/{0}'.format(station_id)
  shutil.rmtree(model_path, ignore_errors=True)
  mlflow.sklearn.save_model( model, model_path)
  # ---------------------------------
  
  # FORECAST
  # ---------------------------------  
  # assemble regressors
  regressors_pd = group_pd[['ds', 'temp_f', 'precip_in']]

  # assemble timeseries
  timeseries_pd = model.make_future_dataframe(
    periods=hours_to_forecast, 
    freq='H'
    )
  
  # merge timeseries with regressors to form forecast dataframe
  future_pd = timeseries_pd.merge(
    regressors_pd,
    how='left',
    on='ds',
    sort=True,
    suffixes=('_l','_r')
    )
  
  # generate forecast
  forecast_pd = model.predict(future_pd)
  # ---------------------------------
  
  # PREPARE RESULTS
  # ---------------------------------
  # merge forecast with history
  results_pd = forecast_pd.merge(
    history_pd[['ds','y']], 
    how='left', 
    on='ds',
    sort=True,
    suffixes=('_l','_r')
   )
 
  # assign station to results
  results_pd['station_id']=station_id
  # ---------------------------------
  
  return results_pd[
      ['station_id', 'ds', 
       'y', 'yhat', 'yhat_lower', 'yhat_upper',
       'trend', 'trend_lower', 'trend_upper', 
       'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper', 
       'daily', 'daily_lower', 'daily_upper',
       'weekly', 'weekly_lower', 'weekly_upper', 
       'yearly', 'yearly_lower', 'yearly_upper', 
       'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
       'holidays', 'holidays_lower', 'holidays_upper']
        ]

# COMMAND ----------

# MAGIC %md As before, we will define and replicate our holidays DataFrame:

# COMMAND ----------

# identify hours that should be treated as aligned with holidays
holidays_pd = spark.sql('''
    SELECT
      b.hour as ds,
      a.holiday as holiday
    FROM citibike.holidays a
    INNER JOIN citibike.periods b
      ON a.date=to_date(b.hour)
    ''').toPandas()

# replicate a copy of the holidays dataset to each node
holidays_broadcast = sc.broadcast(holidays_pd)

# COMMAND ----------

# MAGIC %md And now we can define the dataset and logic for generating the forecasts:

# COMMAND ----------

from pyspark.sql.functions import lit

# define number of hours to forecast
hours_to_forecast = 36

# assemble historical dataset for training
inputs = spark.sql('''
  SELECT
    a.station_id,
    a.hour as ds, 
    COALESCE(b.rentals,0) as y,
    c.avg_temp_f as temp_f,
    COALESCE(c.precip_in,0) as precip_in,
    a.is_historical  
  FROM ( -- all rental hours by currently active stations
    SELECT
      y.station_id,
      x.hour,
      CASE WHEN x.hour <= y.end_date THEN 1 ELSE 0 END as is_historical
    FROM citibike.periods x
    INNER JOIN citibike.stations_most_active y
     ON x.hour BETWEEN y.start_date AND (y.end_date + INTERVAL {0} HOURS)
    ) a
  LEFT OUTER JOIN citibike.rentals b
    ON a.station_id=b.station_id AND a.hour=b.hour
  LEFT OUTER JOIN citibike.weather c
    ON a.hour=c.hour
  '''.format(hours_to_forecast)
  )

# generate forecast
forecast = (
   inputs
    .groupBy('station_id', lit(hours_to_forecast))
    .apply(get_forecast)
  )
forecast.createOrReplaceTempView('forecast_timeseries_with_regressors')

# COMMAND ----------

# MAGIC %md We can now trigger the execution of our logic and load the resulting forecasts to a table for long-term persistence:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.forecast_timeseries_with_regressors;
# MAGIC 
# MAGIC CREATE TABLE citibike.forecast_timeseries_with_regressors 
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT *
# MAGIC   FROM forecast_timeseries_with_regressors;

# COMMAND ----------

# MAGIC %md And now we can release our holidays DataFrame from memory:

# COMMAND ----------

holidays_broadcast.unpersist(blocking=True)

# COMMAND ----------

# MAGIC %md Revisiting the forecast for Station 518, E 39 St & 2 Ave:

# COMMAND ----------

# extract the forecast from our persisted dataset
forecast_pd = (
  spark
    .table('citibike.forecast_timeseries_with_regressors')
    .filter('station_id=518')
    ).toPandas()

# COMMAND ----------

# retrieve the model for this station 
model = mlflow.sklearn.load_model('/dbfs/mnt/citibike/timeseries_regressors/518')

# COMMAND ----------

trends_fig = model.plot_components(forecast_pd)
display(trends_fig)

# COMMAND ----------

from datetime import datetime

# construct a visualization of the forecast
predict_fig = model.plot(forecast_pd, xlabel='hour', ylabel='rentals')

# adjust the x-axis to focus on a limited date range
xlim = predict_fig.axes[0].get_xlim()
new_xlim = (datetime.strptime('2020-01-15','%Y-%m-%d'), datetime.strptime('2020-02-03','%Y-%m-%d'))
predict_fig.axes[0].set_xlim(new_xlim)

# display the chart
display(predict_fig)

# COMMAND ----------

# MAGIC %md Again, we generate our per-station evaluation metrics along with a summary metric for comparison with our other modeling techniques:

# COMMAND ----------

# MAGIC %sql -- per station
# MAGIC SELECT
# MAGIC   e.station_id,
# MAGIC   e.error_sum/n as MAE,
# MAGIC   e.error_sum_abs/n as MAD,
# MAGIC   e.error_sum_sqr/n as MSE,
# MAGIC   POWER(e.error_sum_sqr/n, 0.5) as RMSE,
# MAGIC   e.error_sum_abs_prop_y/n as MAPE
# MAGIC FROM (
# MAGIC   SELECT -- error base values 
# MAGIC     x.station_id,
# MAGIC     COUNT(*) as n,
# MAGIC     SUM(x.yhat-x.y) as error_sum,
# MAGIC     SUM(ABS(x.yhat-x.y)) as error_sum_abs,
# MAGIC     SUM(POWER((x.yhat-x.y),2)) as error_sum_sqr,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.y_corrected)) as error_sum_abs_prop_y,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.yhat)) as error_sum_abs_prop_yhat,
# MAGIC     SUM(x.y) as sum_y,
# MAGIC     SUM(x.yhat) as sum_yhat
# MAGIC   FROM ( -- actuals vs. forecast
# MAGIC     SELECT
# MAGIC       a.station_id,
# MAGIC       a.ds as ds,
# MAGIC       CAST(COALESCE(a.y,0) as float) as y,
# MAGIC       CAST(COALESCE(a.y,1) as float) as y_corrected,
# MAGIC       a.yhat
# MAGIC     FROM citibike.forecast_timeseries_with_regressors a
# MAGIC     INNER JOIN citibike.stations b
# MAGIC       ON a.station_id = b.station_id AND
# MAGIC          a.ds <= b.end_date
# MAGIC      ) x
# MAGIC    GROUP BY x.station_id
# MAGIC   ) e
# MAGIC ORDER BY e.station_id

# COMMAND ----------

# MAGIC %sql -- all stations
# MAGIC 
# MAGIC SELECT
# MAGIC   e.error_sum/n as MAE,
# MAGIC   e.error_sum_abs/n as MAD,
# MAGIC   e.error_sum_sqr/n as MSE,
# MAGIC   POWER(e.error_sum_sqr/n, 0.5) as RMSE,
# MAGIC   e.error_sum_abs_prop_y/n as MAPE
# MAGIC FROM (
# MAGIC   SELECT -- error base values 
# MAGIC     COUNT(*) as n,
# MAGIC     SUM(x.yhat-x.y) as error_sum,
# MAGIC     SUM(ABS(x.yhat-x.y)) as error_sum_abs,
# MAGIC     SUM(POWER((x.yhat-x.y),2)) as error_sum_sqr,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.y_corrected)) as error_sum_abs_prop_y,
# MAGIC     SUM(ABS((x.yhat-x.y)/x.yhat)) as error_sum_abs_prop_yhat,
# MAGIC     SUM(x.y) as sum_y,
# MAGIC     SUM(x.yhat) as sum_yhat
# MAGIC   FROM ( -- actuals vs. forecast
# MAGIC     SELECT
# MAGIC       a.ds as ds,
# MAGIC       CAST(COALESCE(a.y,0) as float) as y,
# MAGIC       CAST(COALESCE(a.y,1) as float) as y_corrected,
# MAGIC       a.yhat
# MAGIC     FROM citibike.forecast_timeseries_with_regressors a
# MAGIC     INNER JOIN citibike.stations b
# MAGIC       ON a.station_id = b.station_id AND
# MAGIC          a.ds <= b.end_date
# MAGIC      ) x
# MAGIC   ) e