import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd

# Create datetime index with 1-minute intervals starting from 01.01.2024 at 13:00 hours
start_time = pd.Timestamp('2024-01-01 13:00:00')
end_time = start_time + pd.Timedelta(minutes=100)
datetime_index = pd.date_range(start=start_time, end=end_time, freq='1Min')

# Create DataFrame with datetime index
df = pd.DataFrame(index=datetime_index)

# Initialize columns with NaN values
df['flow1'] = pd.Series([np.nan] * len(df), index=df.index)
df['temperature1'] = pd.Series([np.nan] * len(df), index=df.index)
df['flow2'] = pd.Series([np.nan] * len(df), index=df.index)
df['temperature2'] = pd.Series([np.nan] * len(df), index=df.index)
df['qdot1'] = pd.Series([np.nan] * len(df), index=df.index)

# Fill the DataFrame according to specified conditions
# Between 13:10 and 13:30, set flow1 = -0.2 and temperature1 = 10°C
df.loc['2024-01-01 13:10:00':'2024-01-01 13:30:00', 'flow1'] = -0.2
df.loc['2024-01-01 13:10:00':'2024-01-01 13:30:00', 'temperature1'] = 10

# Between 13:10 and 13:30, set flow2 = 0.2 and temperature2 = 60°C
df.loc['2024-01-01 13:10:00':'2024-01-01 13:30:00', 'flow2'] = 0.2
df.loc['2024-01-01 13:10:00':'2024-01-01 13:30:00', 'temperature2'] = 60

# Between 13:50 and 14:00, set qdot1 = 1000
df.loc['2024-01-01 13:50:00':'2024-01-01 14:00:00', 'qdot1'] = 1000

print(df)
