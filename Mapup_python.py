#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[54]:


df=pd.read_csv('dataset-1.csv')


# In[5]:


#QUESTION 1:

import pandas as pd

def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Pivot the DataFrame to create a matrix with 'id_1' as index, 'id_2' as columns, and 'car' as values
    car_matrix = df.pivot_table(values='car', index='id_1', columns='id_2', fill_value=0)

    # Set the diagonal values to 0
    for idx in car_matrix.index:
        car_matrix.loc[idx, idx] = 0

    return car_matrix

# Example usage:
# Assuming df is your dataset-1.csv loaded as a DataFrame
# df = pd.read_csv('dataset-1.csv')
# result = generate_car_matrix(df)
# print(result)


# In[7]:


import pandas as pd

def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
    car_matrix = df.pivot_table(values='car', index='id_1', columns='id_2', fill_value=0)

    for idx in car_matrix.index:
        car_matrix.loc[idx, idx] = 0

    return car_matrix

result = generate_car_matrix(df)
print(result)


# In[9]:


#QUEST 2

import pandas as pd

def get_type_count(df: pd.DataFrame) -> dict:
 
    def categorize_car(value):
        if value <= 15:
            return 'low'
        elif 15 < value <= 25:
            return 'medium'
        else:
            return 'high'

    # Create a new column 'car_type' based on the categorization function
    df['car_type'] = df['car'].apply(categorize_car)

    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort:
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


result = get_type_count(df)
print(result)


# In[10]:


#QUEST 3

import pandas as pd

def get_bus_indexes(df: pd.DataFrame) -> list:

    
    bus_mean = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    bus_indexes.sort()

    return bus_indexes

result = get_bus_indexes(df)
print(result)


# In[80]:


#QUEST 4

import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the time data by checking whether the timestamps
    for each unique (id, id_2) pair cover a full 24-hour period and span
    all 7 days of the week.

    Args:
        df (pandas.DataFrame): Input dataframe with columns 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'

    Returns:
        pd.Series: Boolean series indicating if each (id, id_2) pair has incorrect timestamps
    """
    # Map the given dates to days of the week
    day_mapping = {
        'Monday': '2023-01-02',
        'Tuesday': '2023-01-03',
        'Wednesday': '2023-01-04',
        'Thursday': '2023-01-05',
        'Friday': '2023-01-06',
        'Saturday': '2023-01-07',
        'Sunday': '2023-01-08'
    }

    # Combine 'startDay' and 'startTime' into a single datetime column
    df['start_datetime'] = pd.to_datetime(df['startDay'].map(day_mapping) + ' ' + df['startTime'], errors='coerce')

    # Combine 'endDay' and 'endTime' into a single datetime column
    df['end_datetime'] = pd.to_datetime(df['endDay'].map(day_mapping) + ' ' + df['endTime'], errors='coerce')

    # Exclude rows with missing or invalid datetime values
    df = df.dropna(subset=['start_datetime', 'end_datetime'])

    # Group by ('id', 'id_2') and check if the timestamps cover a full 24-hour period and span all 7 days
    result_series = df.groupby(['id', 'id_2']).apply(lambda group: (
        (group['start_datetime'].min().time() == pd.Timestamp('00:00:00').time()) and
        (group['end_datetime'].max().time() == pd.Timestamp('23:59:59').time()) and
        (sorted(group['start_datetime'].dt.weekday.unique()) == list(range(7)))
    ))

    return result_series

# Example usage:
df = df1  # Replace 'path/to/your/dataset-2.csv' with the actual path to your CSV file
result = time_check(df)
print(result)


# In[56]:


#quest 5 : remaining


# ##dataset 2:  task 1 quest 6

# In[106]:


df=pd.read_csv('dataset-2.csv')
df


# In[107]:


import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the time data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.

    Args:
        df (pandas.DataFrame): Input dataframe with columns 'id', 'id_2', and 'timestamp'

    Returns:
        pd.Series: Boolean series indicating if each (`id`, `id_2`) pair has incorrect timestamps
    """
    # Combine 'startDay' and 'startTime' into a single datetime column
    df['start_datetime'] = pd.to_datetime(df1['startDay'] + ' ' + df1['startTime'])

    # Combine 'endDay' and 'endTime' into a single datetime column
    df['end_datetime'] = pd.to_datetime(df1['endDay'] + ' ' + df1['endTime'])

    # Calculate the duration of each timestamp pair
    df1['duration'] = df1['end_datetime'] - df1['start_datetime']

    # Group by ('id', 'id_2') and check if the timestamps cover a full 24-hour and 7 days period
    result_series = df1.groupby(['id', 'id_2']).apply(lambda group: (
        (group['start_datetime'].min().time() == pd.Timestamp('00:00:00').time()) and
        (group['end_datetime'].max().time() == pd.Timestamp('23:59:59').time()) and
        (group['start_datetime'].min().weekday() == 0) and
        (group['end_datetime'].max().weekday() == 6) and
        (group['duration'].sum() >= pd.Timedelta(days=6, hours=23, minutes=59, seconds=59))
    )).droplevel(level=[0, 1])

    return result_series

# Example usage
result = time_check(df1)
print(result)


# In[ ]:


import pandas as pd

def time_check(df):
    """
    Verify the completeness of the time data.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'.

    Returns:
        pd.Series: Boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    # Combine 'startDay' and 'startTime' to create a 'start_timestamp' column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    # Combine 'endDay' and 'endTime' to create an 'end_timestamp' column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Check if each ('id', 'id_2') pair covers a full 24-hour period and spans all 7 days
    completeness_check = df.groupby(['id', 'id_2']).apply(lambda group: 
        (group['start_timestamp'].min().hour == 0) and
        (group['end_timestamp'].max().hour == 23) and
        (group['start_timestamp'].min().day_name() == 'Monday') and
        (group['end_timestamp'].max().day_name() == 'Sunday')
    )

    return completeness_check

# Example usage:
# Assuming df2 is your dataset loaded as a DataFrame
# df2 = pd.read_csv('dataset-2.csv')
result = time_check(df2)
print(result)


# ###dataset3-TASK2

# In[63]:


df3=pd.read_csv('dataset-3.csv')
df3


# In[25]:


pip install geopy


# #Quest 1:

# In[57]:


def calculate_distance_matrix(data):
  """
  This function takes a Pandas DataFrame with columns 'id_start', 'id_end', and 'distance' and returns a DataFrame representing the distances between all pairs of IDs.

  Args:
    data: A Pandas DataFrame with columns 'id_start', 'id_end', and 'distance'.

  Returns:
    A Pandas DataFrame representing the distances between all pairs of IDs.
  """
  # Create a dictionary to store distances between IDs
  distances = {}
  for i in range(len(data)):
    start_id = data.loc[i, 'id_start']
    end_id = data.loc[i, 'id_end']
    distance = data.loc[i, 'distance']
    # Store distances in both directions
    distances[(start_id, end_id)] = distance
    distances[(end_id, start_id)] = distance

  # Create an empty DataFrame to store the distance matrix
  distance_matrix = pd.DataFrame(index=data['id_start'].unique(), columns=data['id_start'].unique())

  # Fill the distance matrix
  for i in range(len(distance_matrix.index)):
    start_id = distance_matrix.index[i]
    for j in range(len(distance_matrix.columns)):
      end_id = distance_matrix.columns[j]
      if start_id == end_id:
        distance_matrix.loc[start_id, end_id] = 0
      else:
        distance_matrix.loc[start_id, end_id] = distances.get((start_id, end_id), None)

  # Ensure the matrix is symmetric
  distance_matrix = distance_matrix.fillna(method='ffill').fillna(method='bfill')

  return distance_matrix

# Create the sample data
data = pd.DataFrame({
  'id_start': [1001400, 1001402, 1001404, 1001406, 1001408, 1001410, 1001412],
  'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414],
  'distance': [9.7, 20.2, 16.0, 21.7, 11.1, 15.6, 18.2]
})

# Calculate and print the distance matrix
distance_matrix = calculate_distance_matrix(data)
print(distance_matrix)


# In[79]:


#QUESTION 2:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
  """
  This function takes a Pandas DataFrame representing a distance matrix and returns a DataFrame with all possible combinations of start and end IDs and their distances.

  Args:
    distance_matrix: A Pandas DataFrame representing a distance matrix.

  Returns:
    A Pandas DataFrame with all possible combinations of start and end IDs and their distances.
  """
  # Create an empty DataFrame to store the unrolled distances
  unrolled_distances = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

  # Loop through the upper triangle of the distance matrix (excluding the diagonal)
  for i in range(len(distance_matrix.index)):
    for j in range(i + 1, len(distance_matrix.columns)):
      start_id = distance_matrix.index[i]
      end_id = distance_matrix.columns[j]
      distance = distance_matrix.loc[start_id, end_id]

      # Add the combination to the unrolled distances DataFrame
      unrolled_distances = unrolled_distances.append({
          'id_start': start_id,
          'id_end': end_id,
          'distance': distance
      }, ignore_index=True)

  return unrolled_distances

# Use the sample distance matrix from Question 1
distance_matrix = pd.DataFrame({
  '1001400': [0, 9.7, 20.2, 45.9, 67.6, 78.7, 94.3],
  '1001402': [9.7, 0, 20.2, 36.2, 57.9, 69, 84.6],
  '1001404': [20.2, 20.2, 0, 16, 37.7, 48.8, 64.4],
  '1001406': [45.9, 36.2, 16, 0, 21.7, 32.8, 48.4],
  '1001408': [67.6, 57.9, 37.7, 21.7, 0, 11.1, 26.7],
  '1001410': [78.7, 69, 48.8, 32.8, 11.1, 0, 15.6],
  '1001412': [94.3, 84.6, 64.4, 48.4, 26.7, 15.6, 0],
})

# Unroll the distance matrix
unrolled_distances = unroll_distance_matrix(distance_matrix)

# Print the first 10 rows of the unrolled distances
print(unrolled_distances.head(10))


# In[99]:


import pandas as pd


def find_ids_within_ten_percentage_threshold(df3, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
        of the reference ID's average distance.
    """

    reference_avg = df3[df3["id_start"] == reference_id]["distance"].mean()
    threshold_min = reference_avg * 0.9
    threshold_max = reference_avg * 1.1

    filtered_df = df3[df3["id_start"] != reference_id]
    filtered_df = filtered_df[filtered_df["distance"].between(threshold_min, threshold_max)]
    filtered_df = filtered_df.groupby("id_start")["distance"].mean().reset_index()
    filtered_df = filtered_df.sort_values(by="id_start")

    return filtered_df[["id_start"]].values.ravel().tolist()

# Create the DataFrame
df = pd.DataFrame({
    "id_start": [1001400, 1001402, 1001404, 1001406, 1001408],
    "id_end": [1001402, 1001404, 1001406, 1001408, 1001410],
    "distance": [9.7, 20.2, 16.0, 21.7, 11.1]
})

# Find IDs within 10% of the average distance of id_start 1001402
reference_id = 1001402
filtered_ids = find_ids_within_ten_percentage_threshold(df.copy(), reference_id)

# Print the filtered IDs
print(filtered_ids)


# In[ ]:





# In[84]:


#QUESTION 4 - TASK2


import pandas as pd


def calculate_toll_rate(df3):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """

    rate_coefficients = {
        "moto": 0.8,
        "car": 1.2,
        "rv": 1.5,
        "bus": 2.2,
        "truck": 3.6
    }

    for vehicle_type, rate in rate_coefficients.items():
        df3[vehicle_type] = df3["distance"] * rate

    return df3

result = calculate_toll_rate(df3)
print(result)


# In[105]:


import pandas as pd

def calculate_time_based_toll_rates(df3):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df3 (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """

    # Check if df3 is a DataFrame
    if not isinstance(df3, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")

    # Check if 'id_start' and 'id_end' columns are present
    if 'id_start' not in df3.columns or 'id_end' not in df3.columns:
        raise ValueError("DataFrame must contain 'id_start' and 'id_end' columns")

    # Define time ranges and discount factors
    time_ranges = {
        # Weekdays
        "00:00-10:00": {
            "start_time": pd.to_datetime("00:00:00").time(),
            "end_time": pd.to_datetime("10:00:00").time(),
            "discount_factor": 0.8
        },
        "10:00-18:00": {
            "start_time": pd.to_datetime("10:00:00").time(),
            "end_time": pd.to_datetime("18:00:00").time(),
            "discount_factor": 1.2
        },
        "18:00-23:59": {
            "start_time": pd.to_datetime("18:00:00").time(),
            "end_time": pd.to_datetime("23:59:59").time(),
            "discount_factor": 0.8
        },
        # Weekends
        "00:00-23:59": {
            "start_time": pd.to_datetime("00:00:00").time(),
            "end_time": pd.to_datetime("23:59:59").time(),
            "discount_factor": 0.7
        }
    }

    # Expand DataFrame with time-based entries
    expanded_df3 = pd.DataFrame()
    for idx, row in df3.iterrows():
        distance = row["distance"]
        
        # Convert 'startDay' to a datetime object
        start_date_str = row["startDay"]
        try:
            start_date = pd.to_datetime(start_date_str, errors='raise')  # This line attempts to convert the string to a Timestamp
        except ValueError:
            print(f"Error converting {start_date_str} to Timestamp for row {idx}")
            continue  # Skip the row if conversion fails

        # Access elements using dot notation
        start_day = start_date.strftime("%A")

        for time_range, details in time_ranges.items():
            start_time = details["start_time"]
            end_time = details["end_time"]
            discount_factor = details["discount_factor"]

            # Calculate toll rates for each vehicle type
            for vehicle_type in ["moto", "car", "rv", "bus", "truck"]:
                toll_rate = row[vehicle_type] * discount_factor
                expanded_df3 = expanded_df3.append({
                    "id_start": row["id_start"],
                    "id_end": row["id_end"],
                    "distance": distance,
                    f"{vehicle_type}_rate": toll_rate,
                    "start_day": start_day,
                    "start_time": start_time,
                    "end_day": start_day,
                    "end_time": end_time
                }, ignore_index=True)

    # Sort DataFrame by start date and time
    expanded_df3 = expanded_df3.sort_values(by=["start_day", "start_time"])

    return expanded_df3

# Example usage:
# Assuming df3 is your dataset loaded as a DataFrame
# df3 = pd.read_csv('your_dataset.csv')
result = calculate_time_based_toll_rates(df3)
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




