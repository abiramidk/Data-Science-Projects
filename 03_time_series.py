# Databricks notebook source
# MAGIC %md ##Forecasting using Time Series Analysis
# MAGIC 
# MAGIC In this notebook, we will develop an hourly forecast for each station using a scale-out pattern described [here](https://databricks.com/blog/2020/01/27/time-series-forecasting-prophet-spark.html). To do this, we will use [Facebook Prophet](https://facebook.github.io/prophet/), a popular timeseries forecasting library.  We will also make use of [mlflow](https://mlflow.org/), a popular framework for the management of machine learning models, to enable the persistence of the models we generate:  

# COMMAND ----------

# install required libraries
dbutils.library.installPyPI('FBProphet', version='0.5') # find latest version of fbprophet here: https://pypi.org/project/fbprophet/
dbutils.library.installPyPI('holidays',version='0.9.12') # this line is in response to this issue with fbprophet 0.5: https://github.com/facebook/prophet/issues/1293
dbutils.library.installPyPI('mlflow')
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md We will now implement a function for the training of models and generating of forecasts on a per-station basis. Notice this function, unlike the functions defined in the previously reference blog post, accepts the keys on which our data is grouped, *i.e.* the station id and the hours for which to produce the forecast.  Notice too that the incoming dataset consists of historical records on which the model will be trained and future records for which forecasts should be generated.  In this specific scenario, this approach is unnecessary, but in later scenarios we will need to structure our data this way so that weather predictions aligned with future periods can more easily be passed into the function.  The inclusion of these data and the subsequent filtering for historical data is therefore simply for consistency purposes with future notebooks:

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
      
  # train model
  model.fit( history_pd )

  # save models for potential later use
  model_path = '/dbfs/mnt/citibike/timeseries/{0}'.format(station_id)
  shutil.rmtree(model_path, ignore_errors=True)
  mlflow.sklearn.save_model( model, model_path)
  # ---------------------------------
  
  # FORECAST
  # ---------------------------------  
  # make forecast dataframe
  future_pd = model.make_future_dataframe(
    periods=hours_to_forecast, 
    freq='H'
   )
  
  # retrieve forecast
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

# MAGIC %md In the function, we are using Prophet as our timeseries learner. This model type has the ability to accomodate holiday information.  When dates are identified as holidays, the values on those dates are not used to train the more general timeseries model and a holiday-specific model is produced to deal with anomolous behavior which may be associated with those points in time.
# MAGIC 
# MAGIC The holiday dataset passed to Prophet is extracted from a variable called *holidays_broadcast* which has yet to be defined.  We'll define that now as a pandas DataFrame deployed as a Spark broadcast variable which makes a replica of the DataFrame available on each worker node.  Managing the variable this way will minimize the amount of data transfer required to send holiday information to our function:

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

# MAGIC %md With everything in place, we can now define the historical dataset from which we will generate our forecasts.
# MAGIC 
# MAGIC Please note there are much easier ways to generate the dataset required for this analysis.  We are writing our query in this manner in preperation for adding weather data to the dataset in our next notebook:

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
    a.is_historical  -- this field will be important when we bring in weather data
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
  '''.format(hours_to_forecast)
  )

# forecast generation logic
forecast = (
  inputs
    .groupBy(inputs.station_id, lit(hours_to_forecast))
    .apply(get_forecast)
  )
forecast.createOrReplaceTempView('forecast_timeseries')

# COMMAND ----------

# MAGIC %md With everything in place, we will now trigger the execution of our logic and load the resulting forecasts to a table for long-term persistence:

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS citibike.forecast_timeseries;
# MAGIC 
# MAGIC CREATE TABLE citibike.forecast_timeseries 
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT *
# MAGIC   FROM forecast_timeseries;

# COMMAND ----------

# MAGIC %md With our training work completed, we can release our holidays DataFrame, currently pinned in the memory of our worker nodes:

# COMMAND ----------

holidays_broadcast.unpersist(blocking=True)

# COMMAND ----------

# MAGIC %md With model training and forecasting completed, let's examine the forecast generated for one of the more popular stations, Station 518 at E 39 St & 2 Ave:

# COMMAND ----------

# extract the station's forecast from our persisted dataset
forecast_pd = (
  spark
    .table('citibike.forecast_timeseries')
    .filter('station_id=518')
    ).toPandas()

# COMMAND ----------

# retrieve the model for this station 
model = mlflow.sklearn.load_model('/dbfs/mnt/citibike/timeseries/518')

# COMMAND ----------

# MAGIC %md Here, we can examine the model components:

# COMMAND ----------

trends_fig = model.plot_components(forecast_pd)
display(trends_fig)

# COMMAND ----------

# MAGIC %md And here we can visualize the forecast:

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

# MAGIC %md Of course, a better way to evaluate the accuracy of our models is through the calculation of evaluation metrics.  We can do this per station or in aggregate across all stations with a little SQL:

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
# MAGIC     FROM citibike.forecast_timeseries a
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
# MAGIC     FROM citibike.forecast_timeseries a
# MAGIC     INNER JOIN citibike.stations b
# MAGIC       ON a.station_id = b.station_id AND
# MAGIC          a.ds <= b.end_date
# MAGIC      ) x
# MAGIC   ) e