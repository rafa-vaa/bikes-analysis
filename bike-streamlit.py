import pandas as pd
import streamlit as st
import joblib
import numpy as np
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots



def descriptive_analysis(data):

    # Descriptive Analysis
    st.subheader("Descriptive Statistics")

    hourly_demand_yearly = data.groupby(['yr', 'hr'])['cnt'].mean().reset_index()
    hourly_demand_yearly_user = data.groupby(['yr', 'hr'])[['registered', 'casual', 'cnt']].mean().reset_index()

    # Filter data for year 0 and year 1
    year_0_data = hourly_demand_yearly[hourly_demand_yearly['yr'] == 0]
    year_1_data = hourly_demand_yearly[hourly_demand_yearly['yr'] == 1]

    year_0_registered_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 0]
    year_1_registered_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 1]

    # Filter data for year 0 and year 1 for casual users
    year_0_casual_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 0]
    year_1_casual_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 1]

    # Calculate the total number of registered and casual users in year 0
    year_0_registered_total = year_0_registered_data['registered'].sum()
    year_0_casual_total = year_0_casual_data['casual'].sum()

    # Calculate the total number of registered and casual users in year 1
    year_1_registered_total = year_1_registered_data['registered'].sum()
    year_1_casual_total = year_1_casual_data['casual'].sum()

    # Calculate the proportion of registered vs casual users in year 0 and year 1
    year_0_proportion_registered = year_0_registered_total / (year_0_registered_total + year_0_casual_total)
    year_0_proportion_casual = year_0_casual_total / (year_0_registered_total + year_0_casual_total)

    year_1_proportion_registered = year_1_registered_total / (year_1_registered_total + year_1_casual_total)
    year_1_proportion_casual = year_1_casual_total / (year_1_registered_total + year_1_casual_total)


    data_yr_season = data.groupby(['yr', 'season'])["cnt"].mean().reset_index()
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data_yr_season[data_yr_season['yr'] == 0]['season'],
        y=data_yr_season[data_yr_season['yr'] == 0]['cnt'],
        name='Year 0',
        marker_color='lightskyblue'
    ))

    fig.add_trace(go.Bar(
        x=data_yr_season[data_yr_season['yr'] == 1]['season'],
        y=data_yr_season[data_yr_season['yr'] == 1]['cnt'],
        name='Year 1',
        marker_color='lightgreen'
    ))

    fig.update_layout(
        title='Average Bike Rental Count by Season for Each Year',
        xaxis=dict(title='Season'),
        yaxis=dict(title='Average Rental Count'),
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)',
        barmode='stack',
        showlegend=True
    )

    st.plotly_chart(fig)


    # Group the data by year and hour, and calculate the average rental count for each hour and year
    hourly_demand_yearly = data.groupby(['yr', 'hr'])['cnt'].mean().reset_index()
    data['is_rush_hour'] = data['hr'].apply(lambda x: 1 if (7 <= x < 9) or (17 <= x < 19) else 0)
    rush_hour_demand = data.groupby(['is_rush_hour', 'yr'])[['registered', 'casual', 'cnt']].mean().reset_index()

    labels = ['Registered', 'Casual']
    year_0_values = [year_0_proportion_registered, year_0_proportion_casual]
    year_1_values = [year_1_proportion_registered, year_1_proportion_casual]

    # Create subplots for each year
    fig = go.Figure()

    fig.add_trace(go.Pie(labels=labels, values=year_0_values, name='Year 0',
                     marker_colors=['limegreen', 'dodgerblue'], hole=.3,
                     domain={'x': [0, 0.5], 'y': [0, 1]}, opacity=0.7))

    # Subplot for year 1
    fig.add_trace(go.Pie(labels=labels, values=year_1_values, name='Year 1',
                     marker_colors=['limegreen', 'dodgerblue'], hole=.3,
                     domain={'x': [0.5, 1], 'y': [0, 1]}, opacity=0.7))


    # Update layout
    fig.update_layout(
        title_text="Proportion of Registered vs Casual Users in Year 0 and Year 1",
        annotations=[dict(text='Year 0', x=0.23, y=0.5, font_size=12, showarrow=False),
                    dict(text='Year 1', x=0.77, y=0.5, font_size=12, showarrow=False)]
    )

    # Show the plot
    st.plotly_chart(fig)

    # temp
    data['temp_range'] = pd.cut(data['temp'], bins=5)
    data_temp = data.groupby('temp_range')['cnt'].mean().reset_index()
    data_temp['temp_range'] = data_temp['temp_range'].astype(str)  # Convert interval values to strings

    # atemp
    data['atemp_range'] = pd.cut(data['atemp'], bins=6)
    data_atemp = data.groupby('atemp_range')['cnt'].mean().reset_index()
    data_atemp['atemp_range'] = data_atemp['atemp_range'].astype(str)  # Convert interval values to strings

    # hum
    data['hum_range'] = pd.cut(data['hum'], bins=5)
    data_htemp = data.groupby('hum_range')['cnt'].mean().reset_index()
    data_htemp['hum_range'] = data_htemp['hum_range'].astype(str)  # Convert interval values to strings

    # windspeed
    data['windspeed_range'] = pd.cut(data['windspeed'], bins=5)
    data_wtemp = data.groupby('windspeed_range')['cnt'].mean().reset_index()
    data_wtemp['windspeed_range'] = data_wtemp['windspeed_range'].astype(str)  # Convert interval values to strings

    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(go.Bar(x=data_temp['temp_range'], y=data_temp['cnt'], marker_color='tomato'), row=1, col=1)
    fig.add_trace(go.Bar(x=data_atemp['atemp_range'], y=data_atemp['cnt'], marker_color='orange'), row=1, col=2)
    fig.add_trace(go.Bar(x=data_htemp['hum_range'], y=data_htemp['cnt'], marker_color='dodgerblue'), row=2, col=1)
    fig.add_trace(go.Bar(x=data_wtemp['windspeed_range'], y=data_wtemp['cnt'], marker_color='chartreuse'), row=2, col=2)

    fig.update_layout(
        xaxis={'categoryorder': 'array', 'categoryarray': data_temp['temp_range'], 'tickangle': 45},
        xaxis2={'categoryorder': 'array', 'categoryarray': data_atemp['atemp_range'], 'tickangle': 45},
        xaxis3={'categoryorder': 'array', 'categoryarray': data_htemp['hum_range'], 'tickangle': 45},
        xaxis4={'categoryorder': 'array', 'categoryarray': data_wtemp['windspeed_range'], 'tickangle': 45},
        xaxis_title='Temperature Range',
        xaxis2_title='Feeling Temperature Range',
        xaxis3_title='Humidity Range',
        xaxis4_title='Wind Speed Range',
        yaxis_title='Average Number of Bikes Rented',
        height=800,
        width=750,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )

    st.plotly_chart(fig)

    data_pivot = data.pivot_table(
        index='mnth', 
        columns='yr', 
        values=['temp', 'atemp', 'hum', 'windspeed'], 
        aggfunc='mean')

    fig = go.Figure()

    for col in data_pivot.columns:
        feature, year = col
        trace = go.Scatter(
            x=data_pivot.index,
            y=data_pivot[col],
            mode='lines+markers',
            name=f'{feature} - Year {year}'
        )
        fig.add_trace(trace)

    fig.update_layout(
        title='Average ambient features for each month of the year for each year',
        xaxis_title='Month',
        yaxis_title='Value'
    )
    st.plotly_chart(fig)

    # Filter data for year 0 and year 1
    year_0_data = hourly_demand_yearly[hourly_demand_yearly['yr'] == 0]
    year_1_data = hourly_demand_yearly[hourly_demand_yearly['yr'] == 1]

    # Create bar plots for each year
    fig_total = go.Figure()

    # Bar plot for year 0
    fig_total.add_trace(go.Bar(
        x=year_0_data['hr'],
        y=year_0_data['cnt'],
        name='Year 0',
        marker_color='lightskyblue'
    ))

    # Bar plot for year 1
    fig_total.add_trace(go.Bar(
        x=year_1_data['hr'],
        y=year_1_data['cnt'],
        name='Year 1',
        marker_color='lightgreen'
    ))

    # Customize the layout
    fig_total.update_layout(
        title='Average Bike Rental Count by Hour of the Day for Each Year',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Average Rental Count'),
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    # Group the data by year, hour, and user type, and calculate the average rental count for each hour, year, and user type
    hourly_demand_yearly_user = data.groupby(['yr', 'hr'])[['registered', 'casual', 'cnt']].mean().reset_index()

    # Filter data for year 0 and year 1 for registered users
    year_0_registered_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 0]
    year_1_registered_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 1]

    # Create bar plots for each year for registered users
    fig_registered = go.Figure()

    # Bar plot for year 0 registered users
    fig_registered.add_trace(go.Bar(
        x=year_0_registered_data['hr'],
        y=year_0_registered_data['registered'],
        name='Year 0 - Registered',
        marker_color='lightskyblue'
    ))

    # Bar plot for year 1 registered users
    fig_registered.add_trace(go.Bar(
        x=year_1_registered_data['hr'],
        y=year_1_registered_data['registered'],
        name='Year 1 - Registered',
        marker_color='lightgreen'
    ))

    # Filter data for year 0 and year 1 for casual users
    year_0_casual_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 0]
    year_1_casual_data = hourly_demand_yearly_user[hourly_demand_yearly_user['yr'] == 1]

    # Bar plot for year 0 casual users
    fig_casual = go.Figure()

    fig_casual.add_trace(go.Bar(
        x=year_0_casual_data['hr'],
        y=year_0_casual_data['casual'],
        name='Year 0 - Casual',
        marker_color='dodgerblue'
    ))

    # Bar plot for year 1 casual users
    fig_casual.add_trace(go.Bar(
        x=year_1_casual_data['hr'],
        y=year_1_casual_data['casual'],
        name='Year 1 - Casual',
        marker_color='limegreen'
    ))

    # Customize the layout
    fig_casual.update_layout(
        title='Average Bike Rental Count by Hour of the Day for Each User Type in Each Year',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Average Rental Count'),
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )

    # Create a button to toggle between total and user type graphs
    fig_buttons = go.Figure()

    fig_buttons.update_layout(
        title='Toggle Between Total and User Type Graphs',
        xaxis=dict(title='Hour of the Day'),
        yaxis=dict(title='Average Rental Count'),
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(
                        args=[{"visible": [True, False, False]}],
                        label="Total",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, True]}],
                        label="User Type",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    fig_buttons.add_trace(go.Bar(
        x=year_0_data['hr'],
        y=year_0_data['cnt'],
        name='Year 0',
        marker_color='lightskyblue',
        visible=True
    ))

    fig_buttons.add_trace(go.Bar(
        x=year_1_data['hr'],
        y=year_1_data['cnt'],
        name='Year 1',
        marker_color='lightgreen',
        visible=True
    ))

    fig_buttons.add_trace(go.Bar(
        x=year_0_registered_data['hr'],
        y=year_0_registered_data['registered'],
        name='Year 0 - Registered',
        marker_color='lightskyblue',
        visible=False
    ))

    fig_buttons.add_trace(go.Bar(
        x=year_1_registered_data['hr'],
        y=year_1_registered_data['registered'],
        name='Year 1 - Registered',
        marker_color='lightgreen',
        visible=False
    ))

    fig_buttons.add_trace(go.Bar(
        x=year_0_casual_data['hr'],
        y=year_0_casual_data['casual'],
        name='Year 0 - Casual',
        marker_color='dodgerblue',
        visible=False
    ))

    fig_buttons.add_trace(go.Bar(
        x=year_1_casual_data['hr'],
        y=year_1_casual_data['casual'],
        name='Year 1 - Casual',
        marker_color='limegreen',
        visible=False
    ))

    # Show the plots
    st.plotly_chart(fig_buttons)

    # Group the data by month and sum the counts for registered and casual users
    monthly_demand = data.groupby('mnth')[['registered', 'casual']].sum()
    # Group the data by month and calculate the average temperature and humidity
    monthly_avg_temp = data.groupby('mnth')['temp'].mean()
    
    fig = go.Figure()

    fig.add_trace(go.Bar(x=monthly_demand.index, y=monthly_demand['registered'], name='Registered', marker=dict(color='lightblue')))
    fig.add_trace(go.Bar(x=monthly_demand.index, y=monthly_demand['casual'], name='Casual', marker=dict(color='lightgreen')))

    fig.add_trace(go.Scatter(x=monthly_avg_temp.index, y=monthly_avg_temp, name='Avg Temperature',
                            mode='lines', line=dict(color='red'), yaxis='y2'))

    # Update the layout with secondary y-axis
    fig.update_layout(title='Monthly Demand Comparison between Registered and Casual Users and the average temperature',
                    xaxis_title='Month',
                    yaxis_title='Count',
                    legend_title='Data Type',
                    barmode='group',
                    yaxis2=dict(title='Value', overlaying='y', side='right'))
    st.plotly_chart(fig)

    # Create a pivot table with the number of bikes rented each month of the year for each year, taking into account if the day is working day or not
    data_pivot = data.pivot_table(
        index='mnth', 
        columns='workingday', 
        values=['registered', 'casual'], 
        aggfunc='mean')

    fig = go.Figure()

    for col in data_pivot.columns:
        feature, workingday = col
        trace = go.Bar(
            x=data_pivot.index,
            y=data_pivot[col],
            name=f'User type {feature}, Workingday {workingday}'
        )
        fig.add_trace(trace)

    fig.update_layout(
        title='Number of bikes rented each month of the year for each year, it is a working day or not',
        xaxis_title='Month',
        yaxis_title='Count',
        barmode='stack'
    )

    # Add button to toggle between casual and registered
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(
                        args=[{"visible": [True, True]}],
                        label="All",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Not working day",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Working day",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    st.plotly_chart(fig)

    fig = go.Figure()

    # Bar plot for year 0
    fig.add_trace(go.Bar(x=rush_hour_demand[rush_hour_demand['yr'] == 0]['is_rush_hour'], y=rush_hour_demand[rush_hour_demand['yr'] == 0]['registered'],
                        name='Registered - Year 0', marker_color='lightskyblue'))
    fig.add_trace(go.Bar(x=rush_hour_demand[rush_hour_demand['yr'] == 0]['is_rush_hour'], y=rush_hour_demand[rush_hour_demand['yr'] == 0]['casual'],
                            name='Casual - Year 0', marker_color='dodgerblue'))

    # Bar plot for year 1
    fig.add_trace(go.Bar(x=rush_hour_demand[rush_hour_demand['yr'] == 1]['is_rush_hour'], y=rush_hour_demand[rush_hour_demand['yr'] == 1]['registered'],
                        name='Registered - Year 1', marker_color='lightgreen'))
    fig.add_trace(go.Bar(x=rush_hour_demand[rush_hour_demand['yr'] == 1]['is_rush_hour'], y=rush_hour_demand[rush_hour_demand['yr'] == 1]['casual'],
                            name='Casual - Year 1', marker_color='limegreen'))

    # Update the layout
    fig.update_layout(title='Average Bike Rental Count for Rush and Non-Rush Hours for Each User Type in Each Year',
                    xaxis_title='Non-Rush Hour (0) vs Rush Hour (1)',
                    yaxis_title='Average Rental Count',
                    legend_title='User Type',
                    barmode='stack')

    # Show the plot
    st.plotly_chart(fig)

    # Extract day from date
    data['day'] = pd.DatetimeIndex(data['dteday']).day

    # Create feature for beginning, middle, or end of month
    data['day_of_month'] = pd.cut(data['day'], bins=[0,10,20,31], labels=['beginning', 'middle', 'end'])

    # Create a graph in plotly that shows the average number of bikes rented for each day of the month, distinguishing between the beginning, middle and end of the month
    data_pivot = data.pivot_table(
        index='day', 
        columns='day_of_month', 
        values=["casual", "registered"], 
        aggfunc='mean')

    data_pivot[('cnt', 'beginning')] = data_pivot[('casual', 'beginning')] + data_pivot[('registered', 'beginning')]
    data_pivot[('cnt', 'middle')] = data_pivot[('casual', 'middle')] + data_pivot[('registered', 'middle')]
    data_pivot[('cnt', 'end')] = data_pivot[('casual', 'end')] + data_pivot[('registered', 'end')]


    fig = go.Figure()

    for col in data_pivot.columns:
        trace = go.Scatter(
            x=data_pivot.index,
            y=data_pivot[col],
            mode='lines+markers',
            name=f'{col}'
        )
        fig.add_trace(trace)

    fig.update_layout(
        title='Average number of bikes rented for each day of the month',
        xaxis_title='Day',
        yaxis_title='Average number of bikes rented'
    )

    # Add button to toggle between casual, registered, and total rentals
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=list([
                    dict(
                        args=[{"visible": [True, True, True]}],
                        label="All",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False, False]}],
                        label="Beginning",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True, False]}],
                        label="Middle",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, False, True]}],
                        label="End",
                        method="update"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
        ]
    )

    st.plotly_chart(fig)

def update_animation_data(frame_index, pred):
  return go.Scatter(
      x = list(range(24)),
      y = pred[frame_index: frame_index + animation_window],
      mode = 'lines+markers',
      marker = dict(color='royalblue'),
      name = 'Predicted Rentals'
  )

animation_window = 3  # Number of hours displayed in the animation frame
fps = 2  # Frames per second for animation


def predictive_analysis(data, model):

    st.title("Bike Rental Predictions - Hourly Trends")

    pred_set = pd.read_csv('bike_features.csv')

    pred_set['dteday'] = pd.to_datetime(data['dteday'])

    start_date = st.date_input("Select a date")
    
    if st.button("Predict"):

        selected_data = pred_set[pred_set['dteday'] == pd.to_datetime(start_date)]

        predictions = model.predict(selected_data.drop('dteday', axis=1))

        # Statistics section
        avg_rental = np.mean(predictions)
        peak_hour = np.argmax(predictions) + 1  # Add 1 for hour index

        # Statistics with columns and key-value pairs
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Rentals")
            st.write(f"{avg_rental:.2f}")
        with col2:
            st.subheader("Peak Rental Hour")
            st.write(f"{peak_hour}")

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=selected_data['hr'], y=predictions, mode='lines+markers', name='Predictions'))

        fig.update_layout(title='Predictions of Bike Rentals by Hour',
                        xaxis_title='Hour of the Day',
                        yaxis_title='Predicted Rental Count')
        
        st.plotly_chart(fig)
 

def conclusions(data):

    import numpy as np
    from sklearn.linear_model import LinearRegression
    import calendar

    st.header("Main Insights")

    def calculate_insights(data):
        # calculated values
        data_yearly = data.groupby('yr')['cnt'].sum()
        data_seasonal = data.groupby('season')['cnt'].mean() * 24
        data_hourly = data.groupby('hr')['cnt'].mean()
        
        # Dynamic Insights
        average_daily_rentals = data['cnt'].mean()
        yearly_growth = ((data_yearly.iloc[1] - data_yearly.iloc[0]) / data_yearly.iloc[0]) * 100
        peak_hour = data_hourly.idxmax()
        peak_hour_rentals = data_hourly[peak_hour]
        peak_hour_over_average = (peak_hour_rentals / average_daily_rentals - 1) * 100
        holiday_rental_difference = ((data[data['holiday'] == 0]['cnt'].mean() / data[data['holiday'] == 1]['cnt'].mean()) - 1) * 100
        user_contribution = data[['casual', 'registered']].sum() / data['cnt'].sum()
        most_active_month = calendar.month_name[data.groupby('mnth')['cnt'].sum().idxmax()]
        
        # Ambient Variables Analysis with dynamic selection
        X = data[['temp', 'windspeed', 'hum']]
        y = data['cnt']
        model = LinearRegression().fit(X, y)
        ambient_coefficients = model.coef_ * 0.1
        
        return {
            "average_daily_rentals": average_daily_rentals * 24,
            "yearly_growth": yearly_growth,
            "season_rentals": data_seasonal.to_dict(),
            "peak_hour": peak_hour,
            "peak_hour_over_average": peak_hour_over_average,
            "holiday_rental_difference": holiday_rental_difference,
            "user_contribution": user_contribution.to_dict(),
            "most_active_month": most_active_month,
            "ambient_coefficients": ambient_coefficients
        }

    insights = calculate_insights(data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Daily Rentals", f"{insights['average_daily_rentals']:.2f}")
        st.metric("Yearly Growth", f"{insights['yearly_growth']:.2f}%", delta_color="inverse")
        st.metric("Peak Hour", f"{insights['peak_hour']}:00", f"{insights['peak_hour_over_average']:.2f}% over average")
    with col2:
        st.metric("Holiday Impact", f"{insights['holiday_rental_difference']:.2f}%", delta_color="off")
        casual_contribution = insights['user_contribution']['casual'] * 100
        registered_contribution = insights['user_contribution']['registered'] * 100
        st.metric("Casual Users Contribution", f"{casual_contribution:.2f}%")
        st.metric("Registered Users Contribution", f"{registered_contribution:.2f}%")
        st.metric("Most Active Month", insights['most_active_month'])

    st.divider()

    st.subheader("Seasonal Demand")
    for season, avg_rentals in insights['season_rentals'].items():
        st.metric(f"Season {season} Average Daily Rentals", f"{avg_rentals:.2f}")

    st.divider()
    
    st.subheader("Influence of Ambient Variables")
    ambient_vars = ['Temperature', 'Wind Speed', 'Humidity']
    for i, coeff in enumerate(insights['ambient_coefficients']):
        st.metric(f"{ambient_vars[i]} influence (per 0.1 unit change)", f"{coeff:.2f}")

    st.divider()

    #Recommendations

    st.header("Business Recommendations")

    st.subheader("Seasonal Promotions and Pricing Strategies")
    st.write("""
    - Introduce seasonal promotions to maximize revenue during peak demand in summer.
    - Offer special packages or discounts for early bookings to attract casual users and convert them into registered users.
    """)

    st.subheader("Optimize Peak Hours near Business Centers")
    st.write("- Ensure adequate bike availability during peak hours, especially around business districts and popular commuting destinations.")

    st.subheader("Adjust Operations Based on Holidays and Unfavorable Weather Conditions")
    st.write("""
    - Scale down operations during major holidays and in anticipation of extreme weather conditions like heatwaves or snow.
    - Conversely, prepare for full operational capacity during clear weather days.
    """)

    st.subheader("Loyalty Programs, Badges, and Promotions")
    st.write("- Encourage registration and frequent use through loyalty programs, badges, and promotions based on usage patterns.")

    st.subheader("Enhance User Experience Based on Ambient Conditions")
    st.write("""
    - Offer amenities such as free water bottles near stations on hot days to improve the user experience.
    - Consider providing weather protection for bikes to encourage usage during less favorable conditions.
    """)

def main():

    data = pd.read_csv('bike-sharing-hourly.csv')

    data.drop(['instant'], axis=1, inplace=True)


    # Title and Introduction
    st.title("Bike Sharing Rental Predictions")
    st.write("This dashboard allows you to explore various factors affecting bike rentals and predict rental counts.")

    
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}

    #Map season to season names in the dataset
    data['season'] = data['season'].map(season_map)

    
    # Get the current working directory
    working_directory = os.getcwd()
    model_path = os.path.join(working_directory, 'model.pkl')


    # Load the model
    model = joblib.load(model_path)


    sections = ['Descriptive Analysis', 'Predictive Analysis', 'Conclusions']

    # Sidebar for User Input
    section = st.sidebar.selectbox("Select analysis", sections, index=0)
    
    if section == 'Descriptive Analysis':
        descriptive_analysis(data)
    elif section == 'Predictive Analysis':
        predictive_analysis(data, model)
    else:
        conclusions(data)


if __name__ == "__main__":
    main()