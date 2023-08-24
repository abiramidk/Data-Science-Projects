# Databricks notebook source
import json
import pandas as pd
import uuid
import random
from random import randrange
from datetime import datetime, timedelta 
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# DBTITLE 1,user journey agg
# MAGIC %sql 
# MAGIC CREATE OR REPLACE TEMP VIEW user_journey_view AS
# MAGIC SELECT
# MAGIC   sub2.infinity_id AS uid,CASE
# MAGIC     WHEN sub2.conversion == 1 then concat('Start > ', sub2.path, ' > Conversion')
# MAGIC     ELSE concat('Start > ', sub2.path, ' > Null')
# MAGIC   END AS path,
# MAGIC   sub2.first_interaction AS first_interaction,
# MAGIC   sub2.last_interaction AS last_interaction,
# MAGIC   sub2.conversion AS conversion,
# MAGIC   sub2.visiting_order AS visiting_order
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       sub.infinity_id AS infinity_id,
# MAGIC       concat_ws(' > ', collect_list(sub.channelgrouping)) AS path,
# MAGIC       element_at(collect_list(sub.channelgrouping), 1) AS first_interaction,
# MAGIC       element_at(collect_list(sub.channelgrouping), -1) AS last_interaction,
# MAGIC       element_at(collect_list(sub.conversion), -1) AS conversion,
# MAGIC       collect_list(sub.visit_order) AS visiting_order
# MAGIC     FROM
# MAGIC       (
# MAGIC         SELECT
# MAGIC           infinity_id,
# MAGIC           channelgrouping,
# MAGIC           dt,
# MAGIC           conversion,
# MAGIC           dense_rank() OVER (
# MAGIC             PARTITION BY infinity_id
# MAGIC             ORDER BY
# MAGIC               dt asc
# MAGIC           ) as visit_order
# MAGIC         FROM
# MAGIC           msci.mta_tny_data_q3_2021
# MAGIC       ) AS sub
# MAGIC     GROUP BY
# MAGIC       sub.infinity_id
# MAGIC   ) AS sub2;

# COMMAND ----------

_ = spark.sql('''
CREATE TABLE IF NOT EXISTS msci.mta_user_journey
USING DELTA
AS
SELECT * FROM user_journey_view''')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from user_journey_view

# COMMAND ----------

# DBTITLE 1,First touch
# MAGIC %sql 
# MAGIC CREATE OR REPLACE TEMP VIEW attribution_view AS
# MAGIC SELECT
# MAGIC   'first_touch' AS attribution_model,
# MAGIC   first_interaction AS channel,
# MAGIC   round(count(*) / (
# MAGIC      SELECT COUNT(*)
# MAGIC      FROM user_journey_view
# MAGIC      WHERE conversion = 1),2) AS attribution_percent
# MAGIC FROM user_journey_view
# MAGIC WHERE conversion = 1
# MAGIC GROUP BY first_interaction
# MAGIC UNION
# MAGIC SELECT
# MAGIC   'last_touch' AS attribution_model,
# MAGIC   last_interaction AS channel,
# MAGIC   round(count(*) /(
# MAGIC       SELECT COUNT(*)
# MAGIC       FROM user_journey_view
# MAGIC       WHERE conversion = 1),2) AS attribution_percent
# MAGIC FROM user_journey_view
# MAGIC WHERE conversion = 1
# MAGIC GROUP BY last_interaction

# COMMAND ----------

_ = spark.sql('''
CREATE TABLE IF NOT EXISTS msci.mta_touch_attribution
USING DELTA
AS
SELECT * FROM attribution_view''')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from msci.mta_touch_attribution

# COMMAND ----------

attribution_pd = spark.table('msci.mta_touch_attribution').toPandas()
 
sns.set(font_scale=1.1)
sns.catplot(x='channel',y='attribution_percent',hue='attribution_model',data=attribution_pd, kind='bar', aspect=2).set_xticklabels(rotation=15)

# COMMAND ----------

