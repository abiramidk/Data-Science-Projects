# Databricks notebook source
from pyspark.sql.types import StringType, ArrayType
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

def get_transition_array(path):
  '''
    This function takes as input a user journey (string) where each state transition is marked by a >. 
    The output is an array that has an entry for each individual state transition.
  '''
  state_transition_array = path.split(">")
  initial_state = state_transition_array[0]
  
  state_transitions = []
  for state in state_transition_array[1:]:
    state_transitions.append(initial_state.strip()+' > '+state.strip())
    initial_state =  state
  
  return state_transitions

# COMMAND ----------

spark.udf.register("get_transition_array", get_transition_array, ArrayType(StringType()))

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW markov_state_transitions AS
# MAGIC SELECT path,
# MAGIC   explode(get_transition_array(path)) as transition,
# MAGIC   1 AS cnt
# MAGIC FROM
# MAGIC msci.mta_user_journey

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from markov_state_transitions

# COMMAND ----------

# MAGIC %md # Construct Transition Probability matrix

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW transition_matrix AS
# MAGIC SELECT
# MAGIC   left_table.start_state,
# MAGIC   left_table.end_state,
# MAGIC   left_table.total_transitions / right_table.total_state_transitions_initiated_from_start_state AS transition_probability
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       transition,
# MAGIC       sum(cnt) total_transitions,
# MAGIC       trim(SPLIT(transition, '>') [0]) start_state,
# MAGIC       trim(SPLIT(transition, '>') [1]) end_state
# MAGIC     FROM
# MAGIC       markov_state_transitions
# MAGIC     GROUP BY
# MAGIC       transition
# MAGIC     ORDER BY
# MAGIC       transition
# MAGIC   ) left_table
# MAGIC   JOIN (
# MAGIC     SELECT
# MAGIC       a.start_state,
# MAGIC       sum(a.cnt) total_state_transitions_initiated_from_start_state
# MAGIC     FROM
# MAGIC       (
# MAGIC         SELECT
# MAGIC           trim(SPLIT(transition, '>') [0]) start_state,
# MAGIC           cnt
# MAGIC         FROM
# MAGIC           markov_state_transitions
# MAGIC       ) AS a
# MAGIC     GROUP BY
# MAGIC       a.start_state
# MAGIC   ) right_table ON left_table.start_state = right_table.start_state
# MAGIC ORDER BY
# MAGIC   end_state DESC

# COMMAND ----------

# DBTITLE 1,validating transition probability matrix
# MAGIC %sql 
# MAGIC SELECT start_state, round(sum(transition_probability),2) as transition_probability_sum 
# MAGIC FROM transition_matrix
# MAGIC GROUP BY start_state

# COMMAND ----------

# DBTITLE 1,displaying trans-prob-matrix
transition_matrix_pd = spark.table('transition_matrix').toPandas()
transition_matrix_pivot = transition_matrix_pd.pivot(index='start_state',columns='end_state',values='transition_probability')

plt.figure(figsize=(15,7))
sns.set(font_scale=1.2)
sns.heatmap(transition_matrix_pivot,cmap='Blues',vmax=0.25,annot=True)

# COMMAND ----------

# MAGIC %md # Total Conversion Probability

# COMMAND ----------

def get_transition_probability_graph(removal_state = "null"):
  '''
  This function calculates a subset of the transition probability graph based on the state to exclude
      removal_state: channel that we want to exclude from our Transition Probability Matrix
  returns subset of the Transition Probability matrix as pandas Dataframe
  '''
  
  transition_probability_pandas_df = None
  
  if removal_state == "null":
    transition_probability_pandas_df = spark.sql('''select
        trim(start_state) as start_state,
        collect_list(end_state) as next_stages,
        collect_list(transition_probability) as next_stage_transition_probabilities
      from
        transition_matrix
      group by
        start_state''').toPandas()
    
  else:
    transition_probability_pandas_df = spark.sql('''select
      sub1.start_state as start_state,
      collect_list(sub1.end_state) as next_stages,
      collect_list(transition_probability) as next_stage_transition_probabilities
      from
      (
        select
          trim(start_state) as start_state,
          case
            when end_state == \"'''+removal_state+'''\" then 'Null'
            else end_state
          end as end_state,
          transition_probability
        from
          transition_matrix
        where
          start_state != \"'''+removal_state+'''\"
      ) sub1 group by sub1.start_state''').toPandas()

  return transition_probability_pandas_df

# COMMAND ----------

transition_probability_pandas_df = get_transition_probability_graph()

# COMMAND ----------

transition_probability_pandas_df

# COMMAND ----------

def calculate_conversion_probability(transition_probability_pandas_df, calculated_state_conversion_probabilities, visited_states, current_state="Start"):
  '''
  This function calculates total conversion probability based on a subset of the transition probability graph
    transition_probability_pandas_df: This is a Dataframe that maps the current state to all probable next stages along with their transition probability
    removal_state: the channel that we want to exclude from our Transition Probability Matrix
    visited_states: set that keeps track of the states that have been visited thus far in our state transition graph.
    current_state: by default the start state for the state transition graph is Start state
  returns conversion probability of current state/channel 
  '''
 
  if current_state=="Conversion":
    return 1.0
  
  elif (current_state=="Null") or (current_state in visited_states):
    return 0.0
  
  elif current_state in calculated_state_conversion_probabilities.keys():
    return calculated_state_conversion_probabilities[current_state]
  
  else:
    visited_states.add(current_state)
    
    current_state_transition_df = transition_probability_pandas_df.loc[transition_probability_pandas_df.start_state==current_state]
    
    next_states = current_state_transition_df.next_stages.to_list()[0]
    next_states_transition_probab = current_state_transition_df.next_stage_transition_probabilities.to_list()[0]
    
    current_state_conversion_probability_arr = []
    
    import copy
    for next_state, next_state_tx_probability in zip(next_states, next_states_transition_probab):
      current_state_conversion_probability_arr.append(next_state_tx_probability * calculate_conversion_probability(transition_probability_pandas_df, calculated_state_conversion_probabilities, copy.deepcopy(visited_states), next_state))
    
    calculated_state_conversion_probabilities[current_state] =  sum(current_state_conversion_probability_arr)
    
    return calculated_state_conversion_probabilities[current_state]

# COMMAND ----------

total_conversion_probability = calculate_conversion_probability(transition_probability_pandas_df, {}, visited_states=set(), current_state="Start")

# COMMAND ----------

total_conversion_probability

# COMMAND ----------

# MAGIC %md ## Removal effect to calculate attribution

# COMMAND ----------

removal_effect_per_channel = {}
for channel in transition_probability_pandas_df.start_state.to_list():
  if channel!="Start":
    transition_probability_subset_pandas_df = get_transition_probability_graph(removal_state=channel)
    new_conversion_probability =  calculate_conversion_probability(transition_probability_subset_pandas_df, {}, visited_states=set(), current_state="Start")
    removal_effect_per_channel[channel] = round(((total_conversion_probability-new_conversion_probability)/total_conversion_probability), 2)

# COMMAND ----------

conversion_attribution={}

for channel in removal_effect_per_channel.keys():
  conversion_attribution[channel] = round(removal_effect_per_channel[channel] / sum(removal_effect_per_channel.values()), 2)

channels = list(conversion_attribution.keys())
conversions = list(conversion_attribution.values())

conversion_pandas_df= pd.DataFrame({'attribution_model': 
                                    ['markov_chain' for _ in range(len(channels))], 
                                    'channel':channels, 
                                    'attribution_percent': conversions})


# COMMAND ----------

sparkDF=spark.createDataFrame(conversion_pandas_df) 
sparkDF.createOrReplaceTempView("markov_chain_attribution_update")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from markov_chain_attribution_update

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO msci.mta_touch_attribution
# MAGIC USING markov_chain_attribution_update
# MAGIC ON markov_chain_attribution_update.attribution_model = msci.mta_touch_attribution.attribution_model AND markov_chain_attribution_update.channel = msci.mta_touch_attribution.channel
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED
# MAGIC   THEN INSERT *

# COMMAND ----------

# MAGIC %md ## Compare heuristic vs datadriven touch attribution

# COMMAND ----------

attribution_pd = spark.table('msci.mta_touch_attribution').toPandas()

sns.set(font_scale=1.1)
sns.catplot(x='channel',y='attribution_percent',hue='attribution_model',data=attribution_pd, kind='bar', aspect=3)

# COMMAND ----------

