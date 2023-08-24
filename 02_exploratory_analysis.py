# Databricks notebook source
# MAGIC %md ##Exploring the Rental Data
# MAGIC 
# MAGIC With our data prepared, we can now begin exploring it to understand the patterns it contains.  Let's start by examining the number of rentals occuring in each year: 

# COMMAND ----------

# DBTITLE 1,Total Rentals by Year
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT
# MAGIC   year(hour) as year,
# MAGIC   SUM(rentals) as rentals
# MAGIC FROM citibike.rentals
# MAGIC GROUP BY year(hour)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md There has been a steady increase in the number of rentals since the start of the program mid-2013.  But how much of this is tied to the expansion of the network?

# COMMAND ----------

# DBTITLE 1,Total Stations by Year
# MAGIC %sql
# MAGIC SELECT
# MAGIC   year(hour) as year,
# MAGIC   COUNT(DISTINCT station_id) as active_stations
# MAGIC FROM citibike.rentals
# MAGIC GROUP BY year(hour)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md There is a steady uptick in the number of stations that have come online in each year of the program. If we normalize our rentals data by the number of active stations in a given year (and exclude those years for which we have only partial periods of reporting), we can get a better sense of growth trends we might be experiencing at a per-station level, the level at which we intend to perform our forecasting:

# COMMAND ----------

# DBTITLE 1,Rentals per Station by Year
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   x.year,
# MAGIC   x.rentals / x.stations as per_station_rentals
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     year(hour) as year,
# MAGIC     COUNT(DISTINCT station_id) as stations,
# MAGIC     SUM(rentals) as rentals
# MAGIC   FROM citibike.rentals
# MAGIC   WHERE year(hour) BETWEEN 2014 AND 2019
# MAGIC   GROUP BY year(hour)
# MAGIC   ) x
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md From this we may conclude the trend in rentals at a per-station level has a slight upward trend (over the last few years) that's independent from the number of new stations coming online.
# MAGIC 
# MAGIC Now, let's examine seasonal trends in rentals, starting at the month level:

# COMMAND ----------

# DBTITLE 1,Rentals per Station by Month of Year
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT
# MAGIC   MONTH(hour) as month,
# MAGIC   SUM(rentals) / COUNT(DISTINCT station_id) as per_station_rentals
# MAGIC FROM citibike.rentals
# MAGIC GROUP BY MONTH(hour)
# MAGIC ORDER BY month

# COMMAND ----------

# MAGIC %md It appears that rentals pickup in the Spring, continue rising into the Summer, and then begin declining into the Fall with a low-point occuring during the Winter.  This might suggest a correlation with temperature.  Visually comparing the graph for temperature, we can see some similarities between the graph for monthly ridership and max monthly temperature:

# COMMAND ----------

# DBTITLE 1,Avg Monthly Max Temperature
# MAGIC %sql 
# MAGIC SELECT
# MAGIC   x.month,
# MAGIC   AVG(x.max_temp_f) as avg_max_temp_f
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     MONTH(hour) as month,
# MAGIC     YEAR(hour) as year,
# MAGIC     MAX(max_temp_f) as max_temp_f
# MAGIC   FROM citibike.weather
# MAGIC   GROUP BY MONTH(hour), YEAR(hour)
# MAGIC   ) x
# MAGIC GROUP BY x.month
# MAGIC ORDER BY x.month

# COMMAND ----------

# MAGIC %md If we assume riders prefer not to be exposed to the elements during incliment weather, we might expect to see some alignment between ridership and precipitation as well:

# COMMAND ----------

# DBTITLE 1,Avg Monthly Precipitation
# MAGIC %sql 
# MAGIC SELECT
# MAGIC   x.month,
# MAGIC   AVG(x.precip_in) as avg_precip_in
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     MONTH(hour) as month,
# MAGIC     YEAR(hour) as year,
# MAGIC     SUM(precip_in) as precip_in
# MAGIC   FROM citibike.weather
# MAGIC   GROUP BY MONTH(hour), YEAR(hour)
# MAGIC   ) x
# MAGIC GROUP BY x.month
# MAGIC ORDER BY x.month

# COMMAND ----------

# MAGIC %md While the idea that individuals might avoid riding bikes during precipitation events makes intuitive sense, at a monthly level of granularity, the pattern doesn't appear to be present.  This doesn't mean that there is no relationship, simply that one is not clearly observable at this level of aggregation.
# MAGIC 
# MAGIC Returning to our annual seasonal patterns, we can see that the annual seasonal pattern in ridership with a peak in the late Summer months and a valley during the Winter seems to hold up year over year with some variability:

# COMMAND ----------

# DBTITLE 1,Rentals per Station by Month & Year
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT
# MAGIC   TRUNC(hour, 'MM') as month,
# MAGIC   SUM(rentals) / COUNT(DISTINCT station_id) as per_station_rentals
# MAGIC FROM citibike.rentals
# MAGIC GROUP BY TRUNC(hour, 'MM')
# MAGIC ORDER BY month

# COMMAND ----------

# MAGIC %md Moving from an annual to a weekly examination of seasonality, we can see an interesting pattern in rentals over the course of a week.  The weekend days in general have lower ridership than on other days. Weekday rentals are higher with a peak on Wednesdays.  This might suggest heavier use by commuters:

# COMMAND ----------

# DBTITLE 1,Rentals per Station by Day of Week
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT
# MAGIC   day_of_week,
# MAGIC   rentals/stations as per_station_rentals
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     EXTRACT(dayofweek from hour) as day_of_week,
# MAGIC     SUM(rentals) as rentals,
# MAGIC     COUNT(DISTINCT station_id) as stations
# MAGIC   FROM citibike.rentals
# MAGIC   WHERE YEAR(hour) BETWEEN 2014 AND 2020
# MAGIC   GROUP BY EXTRACT(dayofweek from hour)
# MAGIC   ) x
# MAGIC ORDER BY day_of_week

# COMMAND ----------

# MAGIC %md An examination of the daily seasonal patterns more clearly makes the case for heavy commuter use of the bike share program.  On weekdays, there are pronounced spikes in rentals during rush hours(though on Fridays it does appear people are trying to get out of the office a bit earlier than on other weekdays).  On weekends, there's a very different, much more leisurely pattern of utilization in the data: 

# COMMAND ----------

# DBTITLE 1,Hourly Rentals per Station by Day of Week
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   day_of_week,
# MAGIC   hour_of_day,
# MAGIC   rentals/stations as per_station_rentals
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     EXTRACT(dayofweek from hour) as day_of_week,
# MAGIC     HOUR(hour) as hour_of_day,
# MAGIC     SUM(rentals) as rentals,
# MAGIC     COUNT(DISTINCT station_id) as stations
# MAGIC   FROM citibike.rentals
# MAGIC   GROUP BY EXTRACT(dayofweek from hour), HOUR(hour)
# MAGIC   ) x
# MAGIC ORDER BY day_of_week

# COMMAND ----------

# MAGIC %md So how do these hourly patterns hold up on holidays?  While one could argue that holidays appear to adopt the patterns more typical of weekends, the lower frequency of holidays gives the curves a much more erratic shape:

# COMMAND ----------

# DBTITLE 1,Hourly Rentals per Station for Holidays 
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   day_of_week,
# MAGIC   hour_of_day,
# MAGIC   is_holiday,
# MAGIC   rentals/stations as per_station_rentals,
# MAGIC   rentals/stations/holiday_occurances as per_station_rentals_normalized
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CASE WHEN b.date IS NULL THEN 0
# MAGIC     ELSE 1 END as is_holiday,
# MAGIC     EXTRACT(dayofweek from a.hour) as day_of_week,
# MAGIC     HOUR(a.hour) as hour_of_day,
# MAGIC     SUM(a.rentals) as rentals,
# MAGIC     COUNT(DISTINCT a.station_id) as stations,
# MAGIC     COUNT(DISTINCT b.date) as holiday_occurances
# MAGIC   FROM citibike.rentals a
# MAGIC   LEFT OUTER JOIN citibike.holidays b
# MAGIC     ON TO_DATE(a.hour) = TO_DATE(b.date) 
# MAGIC   GROUP BY 
# MAGIC     CASE WHEN b.date IS NULL THEN 0
# MAGIC     ELSE 1 END, 
# MAGIC     EXTRACT(dayofweek from a.hour),
# MAGIC     HOUR(a.hour)
# MAGIC   ) x
# MAGIC WHERE is_holiday = 1
# MAGIC ORDER BY day_of_week, hour_of_day

# COMMAND ----------

# MAGIC %md As we examine the data at an hourly level, it's important to ask, how frequently do we encounter hours of the day within which no bikes are rented? When examined in aggregate, it appears that we have a healthy stream of rentals taking place around the clock, but at an hourly level, many stations have a high number of inactive periods:

# COMMAND ----------

# DBTITLE 1,Percent Periods with No Rentals (1 hour periods)
# MAGIC %sql
# MAGIC SELECT
# MAGIC   a.station_id,
# MAGIC   100 * COUNT(*) /
# MAGIC   SUM(COALESCE(b.rentals, 1)) as Percent_Hours_With_No_Rentals
# MAGIC FROM ( -- all rental hours by currently active stations
# MAGIC   SELECT
# MAGIC     y.station_id,
# MAGIC     x.hour
# MAGIC   FROM citibike.periods x
# MAGIC   INNER JOIN citibike.stations_active y
# MAGIC     ON x.hour BETWEEN y.start_date AND y.end_date
# MAGIC   ) a
# MAGIC LEFT OUTER JOIN citibike.rentals b
# MAGIC   ON a.station_id=b.station_id AND a.hour=b.hour
# MAGIC GROUP BY a.station_id
# MAGIC ORDER BY Percent_Hours_with_No_Rentals

# COMMAND ----------

# MAGIC %md Forecasting periods with no activity is challenging.  If we aggregated our data in 4 hour or 6 hour intervals, how does this pattern change?:

# COMMAND ----------

# DBTITLE 1,Percent Periods with No Rentals (4 hour periods)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.station_id,
# MAGIC   100 * COUNT(*) /
# MAGIC   SUM(COALESCE(b.rentals, 1)) as Percent_Hours_With_No_Rentals
# MAGIC FROM ( -- all rental hours by currently active stations
# MAGIC   SELECT DISTINCT
# MAGIC     y.station_id,
# MAGIC     x.hour_agg
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       hour as hour,
# MAGIC       CAST(
# MAGIC         CAST(CAST(TO_DATE(hour) as timestamp) as double) + 
# MAGIC         (CAST(CAST(EXTRACT(hour from hour)/4 as int) * 4 as double) * 3600)
# MAGIC         as timestamp
# MAGIC         ) as hour_agg
# MAGIC     FROM citibike.periods
# MAGIC     ) x
# MAGIC   INNER JOIN citibike.stations_active y
# MAGIC     ON x.hour BETWEEN y.start_date AND y.end_date
# MAGIC   ) a
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT 
# MAGIC     x.station_id,
# MAGIC     y.hour_agg,
# MAGIC     SUM(x.rentals) as rentals
# MAGIC   FROM citibike.rentals x 
# MAGIC   INNER JOIN (
# MAGIC     SELECT 
# MAGIC       hour as hour,
# MAGIC       CAST(
# MAGIC         CAST(CAST(TO_DATE(hour) as timestamp) as double) + 
# MAGIC         (CAST(CAST(EXTRACT(hour from hour)/4 as int) * 4 as double) * 3600)
# MAGIC         as timestamp
# MAGIC         ) as hour_agg
# MAGIC     FROM citibike.periods
# MAGIC     ) y
# MAGIC     ON x.hour=y.hour
# MAGIC   GROUP BY x.station_id, y.hour_agg
# MAGIC   ) b
# MAGIC   ON a.station_id=b.station_id AND a.hour_agg=b.hour_agg
# MAGIC GROUP BY a.station_id
# MAGIC ORDER BY Percent_Hours_With_No_Rentals

# COMMAND ----------

# DBTITLE 1,Percent Periods with No Rentals (6 hour periods)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.station_id,
# MAGIC   100 * COUNT(*) /
# MAGIC   SUM(COALESCE(b.rentals, 1)) as Percent_Hours_With_No_Rentals
# MAGIC FROM ( -- all rental hours by currently active stations
# MAGIC   SELECT DISTINCT
# MAGIC     y.station_id,
# MAGIC     x.hour_agg
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       hour as hour,
# MAGIC       CAST(
# MAGIC         CAST(CAST(TO_DATE(hour) as timestamp) as double) + 
# MAGIC         (CAST(CAST(EXTRACT(hour from hour)/6 as int) * 6 as double) * 3600)
# MAGIC         as timestamp
# MAGIC         ) as hour_agg
# MAGIC     FROM citibike.periods
# MAGIC     ) x
# MAGIC   INNER JOIN citibike.stations_active y
# MAGIC     ON x.hour BETWEEN y.start_date AND y.end_date
# MAGIC   ) a
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT 
# MAGIC     x.station_id,
# MAGIC     y.hour_agg,
# MAGIC     SUM(x.rentals) as rentals
# MAGIC   FROM citibike.rentals x 
# MAGIC   INNER JOIN (
# MAGIC     SELECT 
# MAGIC       hour as hour,
# MAGIC       CAST(
# MAGIC         CAST(CAST(TO_DATE(hour) as timestamp) as double) + 
# MAGIC         (CAST(CAST(EXTRACT(hour from hour)/6 as int) * 6 as double) * 3600)
# MAGIC         as timestamp
# MAGIC         ) as hour_agg
# MAGIC     FROM citibike.periods
# MAGIC     ) y
# MAGIC     ON x.hour=y.hour
# MAGIC   GROUP BY x.station_id, y.hour_agg
# MAGIC   ) b
# MAGIC   ON a.station_id=b.station_id AND a.hour_agg=b.hour_agg
# MAGIC GROUP BY a.station_id
# MAGIC ORDER BY Percent_Hours_With_No_Rentals

# COMMAND ----------

# MAGIC %md As the size of the intervals increases, the proportion of stations with periods of no activity diminishes.  But so do some of the daily seasonal patterns we explored earlier:

# COMMAND ----------

# DBTITLE 1,Hourly Rentals per Station by Day of Week (4 hour periods)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   day_of_week,
# MAGIC   hour_of_day,
# MAGIC   rentals/stations as per_station_rentals
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     EXTRACT(dayofweek from hour_agg) as day_of_week,
# MAGIC     HOUR(hour_agg) as hour_of_day,
# MAGIC     SUM(rentals) as rentals,
# MAGIC     COUNT(DISTINCT station_id) as stations
# MAGIC   FROM (  
# MAGIC     SELECT 
# MAGIC       x.station_id,
# MAGIC       y.hour_agg,
# MAGIC       SUM(x.rentals) as rentals
# MAGIC     FROM citibike.rentals x 
# MAGIC     INNER JOIN (
# MAGIC       SELECT 
# MAGIC         hour as hour,
# MAGIC         CAST(
# MAGIC           CAST(CAST(TO_DATE(hour) as timestamp) as double) + 
# MAGIC           (CAST(CAST(EXTRACT(hour from hour)/4 as int) * 4 as double) * 3600)
# MAGIC           as timestamp
# MAGIC           ) as hour_agg
# MAGIC       FROM citibike.periods
# MAGIC       ) y
# MAGIC       ON x.hour=y.hour
# MAGIC     GROUP BY x.station_id, y.hour_agg
# MAGIC     ) a
# MAGIC   GROUP BY EXTRACT(dayofweek from hour_agg), HOUR(hour_agg)
# MAGIC   ) x
# MAGIC ORDER BY day_of_week

# COMMAND ----------

# DBTITLE 1,Hourly Rentals per Station by Day of Week (6 hour periods)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   day_of_week,
# MAGIC   hour_of_day,
# MAGIC   rentals/stations as per_station_rentals
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     EXTRACT(dayofweek from hour_agg) as day_of_week,
# MAGIC     HOUR(hour_agg) as hour_of_day,
# MAGIC     SUM(rentals) as rentals,
# MAGIC     COUNT(DISTINCT station_id) as stations
# MAGIC   FROM (  
# MAGIC     SELECT 
# MAGIC       x.station_id,
# MAGIC       y.hour_agg,
# MAGIC       SUM(x.rentals) as rentals
# MAGIC     FROM citibike.rentals x 
# MAGIC     INNER JOIN (
# MAGIC       SELECT 
# MAGIC         hour as hour,
# MAGIC         CAST(
# MAGIC           CAST(CAST(TO_DATE(hour) as timestamp) as double) + 
# MAGIC           (CAST(CAST(EXTRACT(hour from hour)/6 as int) * 6 as double) * 3600)
# MAGIC           as timestamp
# MAGIC           ) as hour_agg
# MAGIC       FROM citibike.periods
# MAGIC       ) y
# MAGIC       ON x.hour=y.hour
# MAGIC     GROUP BY x.station_id, y.hour_agg
# MAGIC     ) a
# MAGIC   GROUP BY EXTRACT(dayofweek from hour_agg), HOUR(hour_agg)
# MAGIC   ) x
# MAGIC ORDER BY day_of_week

# COMMAND ----------

# MAGIC %md And so this is where we need to make a decision on how to proceed.  Aggregating the data to higher-levels of time helps reduce the occurance of zero-valued periods, but there are many stations which appear to have a large portion of inactive periods. Aggregation won't help us avoid these.  
# MAGIC 
# MAGIC For the purposes of this exercise, we'll stick to the hourly level of granularity but limit our analysis to the top 200 most active stations.  The selection of 200 is completely arbitrary but does allow us to focus on our most dynamic stations, all of which still have occassional periods of inactivity.
# MAGIC 
# MAGIC To facilitate this work, we'll create a table identifying these stations.  The following cell is included at the bottom of our data preparation notebook just so that it's not missed should you be preparing to recreate these results:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS citibike.stations_most_active;
# MAGIC 
# MAGIC CREATE TABLE citibike.stations_most_active 
# MAGIC   USING DELTA 
# MAGIC   AS
# MAGIC     SELECT
# MAGIC       a.station_id,
# MAGIC       a.start_date,
# MAGIC       a.end_date,
# MAGIC       100 * COUNT(*) /
# MAGIC       SUM(COALESCE(b.rentals, 1)) as Percent_Hours_With_No_Rentals
# MAGIC     FROM ( -- all rental hours by currently active stations
# MAGIC       SELECT
# MAGIC         y.station_id,
# MAGIC         y.start_date,
# MAGIC         y.end_date,
# MAGIC         x.hour
# MAGIC       FROM citibike.periods x
# MAGIC       INNER JOIN citibike.stations_active y
# MAGIC         ON x.hour BETWEEN y.start_date AND y.end_date
# MAGIC       ) a
# MAGIC     LEFT OUTER JOIN citibike.rentals b
# MAGIC       ON a.station_id=b.station_id AND a.hour=b.hour
# MAGIC     GROUP BY a.station_id, a.start_date, a.end_date
# MAGIC     ORDER BY Percent_Hours_with_No_Rentals
# MAGIC     LIMIT 200;

# COMMAND ----------

