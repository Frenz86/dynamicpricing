import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

# ---- Data Generation ----
def generate_sample_data(n_weeks=52):
    """Generate sample shipping data with seasonal patterns."""
    dates = pd.date_range(start='2023-01-01', periods=n_weeks, freq='W')
    routes = ['ASIA-EUR', 'EUR-USA', 'ASIA-USA']
    container_sizes = ['20ft', '40ft']
    
    data = []
    np.random.seed(42)
    
    for route in routes:
        for size in container_sizes:
            base_price = np.random.uniform(1000, 3000)
            base_volume = np.random.uniform(100, 500)
            
            for date in dates:
                # Add seasonal variation
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
                # Add random variation to price
                price = base_price * seasonal_factor * (1 + np.random.uniform(-0.2, 0.2))
                # Volume responds to price changes
                price_effect = -0.3 * (price - base_price) / base_price
                volume = base_volume * seasonal_factor * (1 + price_effect + np.random.uniform(-0.1, 0.1))
                
                data.append({
                    'date': date,
                    'route': route,
                    'container_size': size,
                    'price': round(price, 2),
                    'volume': round(volume, 0),
                    'revenue': round(price * volume, 2)
                })
    
    return pd.DataFrame(data)

# ---- Elasticity Analysis ----
def calculate_elasticities(df):
    """Calculate price elasticity for each route-size combination."""
    elasticities = {}
    
    for route in df['route'].unique():
        for size in df['container_size'].unique():
            mask = (df['route'] == route) & (df['container_size'] == size)
            route_data = df[mask].copy()
            
            # Calculate percentage changes
            price_pct_change = route_data['price'].pct_change()
            volume_pct_change = route_data['volume'].pct_change()
            
            # Filter valid data points
            valid_mask = (abs(price_pct_change) > 0.001) & ~np.isinf(volume_pct_change/price_pct_change)
            if valid_mask.sum() > 0:
                point_elasticities = volume_pct_change[valid_mask] / price_pct_change[valid_mask]
                point_elasticities = point_elasticities[abs(point_elasticities) < 10]
                elasticity = point_elasticities.median()
            else:
                elasticity = -0.3
            
            # Clip elasticity to reasonable range
            elasticity = np.clip(elasticity, -2.0, -0.1)
            elasticities[(route, size)] = round(elasticity, 3)
    
    return elasticities

# ---- Impact Analysis ----
def calculate_impact(df, route, size, price_change_pct, elasticity):
    """Calculate the impact of price changes on volume and revenue."""
    mask = (df['route'] == route) & (df['container_size'] == size)
    base_data = df[mask].iloc[-1]
    
    base_price = base_data['price']
    base_volume = base_data['volume']
    
    # Calculate new values
    price_change_pct = price_change_pct / 100
    new_price = base_price * (1 + price_change_pct)
    volume_change_pct = elasticity * price_change_pct
    volume_change_pct = np.clip(volume_change_pct, -0.5, 0.5)
    new_volume = base_volume * (1 + volume_change_pct)
    
    base_revenue = base_price * base_volume
    new_revenue = new_price * new_volume
    revenue_change_pct = ((new_revenue / base_revenue) - 1) * 100
    
    return {
        'base_price': base_price,
        'new_price': new_price,
        'base_volume': base_volume,
        'new_volume': new_volume,
        'base_revenue': base_revenue,
        'new_revenue': new_revenue,
        'revenue_change_pct': revenue_change_pct,
        'elasticity': elasticity
    }

# ---- Optimization Analysis ----
def calculate_optimal_price(base_price, base_volume, elasticity):
    """Calculate the optimal price based on elasticity."""
    if elasticity >= -1:
        return base_price * 1.5  # If inelastic, increase price
    else:
        optimal_markup = -1 / elasticity
        return base_price * (1 + optimal_markup)

def generate_demand_curve(base_price, base_volume, elasticity, points=50):
    """Generate points for demand and revenue curves."""
    price_range = np.linspace(base_price * 0.5, base_price * 1.5, points)
    volumes = []
    revenues = []
    
    for p in price_range:
        price_change = (p - base_price) / base_price
        volume_change = elasticity * price_change
        volume = base_volume * (1 + volume_change)
        volumes.append(volume)
        revenues.append(p * volume)
        
    return price_range, volumes, revenues

# ---- Seasonal Analysis ----
def analyze_seasonality(df, route, size):
    """Analyze seasonal patterns in price and volume data."""
    mask = (df['route'] == route) & (df['container_size'] == size)
    data = df[mask].copy()
    data.set_index('date', inplace=True)
    
    # Perform seasonal decomposition
    price_decomp = seasonal_decompose(data['price'], period=12)
    volume_decomp = seasonal_decompose(data['volume'], period=12)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Seasonality', 'Price Trend',
                       'Volume Seasonality', 'Volume Trend')
    )
    
    # Add price components
    fig.add_trace(
        go.Scatter(x=data.index, y=price_decomp.seasonal,
                  name='Price Seasonal', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=price_decomp.trend,
                  name='Price Trend', line=dict(color='red')),
        row=1, col=2
    )
    
    # Add volume components
    fig.add_trace(
        go.Scatter(x=data.index, y=volume_decomp.seasonal,
                  name='Volume Seasonal', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=volume_decomp.trend,
                  name='Volume Trend', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Seasonal Analysis", title_x=0.5)
    
    # Calculate seasonal statistics
    seasonal_stats = {
        'price_seasonality': {
            'peak_month': data.groupby(data.index.month)['price'].mean().idxmax(),
            'trough_month': data.groupby(data.index.month)['price'].mean().idxmin(),
            'seasonal_amplitude': price_decomp.seasonal.max() - price_decomp.seasonal.min()
        },
        'volume_seasonality': {
            'peak_month': data.groupby(data.index.month)['volume'].mean().idxmax(),
            'trough_month': data.groupby(data.index.month)['volume'].mean().idxmin(),
            'seasonal_amplitude': volume_decomp.seasonal.max() - volume_decomp.seasonal.min()
        }
    }
    
    return fig, seasonal_stats

# ---- Forecasting ----
def forecast_future_values(df, route, size, forecast_periods=12):
    """Generate forecasts for price and volume."""
    mask = (df['route'] == route) & (df['container_size'] == size)
    data = df[mask].copy()
    
    # Prepare features
    data['month'] = data['date'].dt.month
    data['trend'] = np.arange(len(data))
    
    # Create and fit models
    price_model = LinearRegression()
    volume_model = LinearRegression()
    
    X = data[['month', 'trend']]
    price_model.fit(X, data['price'])
    volume_model.fit(X, data['volume'])
    
    # Generate future dates
    last_date = data['date'].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=7),
        periods=forecast_periods,
        freq='W'
    )
    
    # Create future features
    future_X = pd.DataFrame({
        'month': future_dates.month,
        'trend': np.arange(len(data), len(data) + forecast_periods)
    })
    
    # Generate predictions
    price_forecast = price_model.predict(future_X)
    volume_forecast = volume_model.predict(future_X)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Forecast', 'Volume Forecast')
    )
    
    # Plot historical and forecasted prices
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['price'],
                  name='Historical Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=future_dates, y=price_forecast,
                  name='Forecasted Price',
                  line=dict(dash='dash', color='blue')),
        row=1, col=1
    )
    
    # Plot historical and forecasted volumes
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['volume'],
                  name='Historical Volume', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=future_dates, y=volume_forecast,
                  name='Forecasted Volume',
                  line=dict(dash='dash', color='green')),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text="12-Week Forecast", title_x=0.5)
    
    # Prepare forecast data
    forecast_data = pd.DataFrame({
        'date': future_dates,
        'forecasted_price': price_forecast,
        'forecasted_volume': volume_forecast
    })
    
    return fig, forecast_data

# ---- Route Comparison ----
def compare_routes(df, size):
    """Compare performance across different routes."""
    data = df[df['container_size'] == size].copy()
    
    # Calculate metrics by route
    route_metrics = data.groupby('route').agg({
        'price': ['mean', 'std'],
        'volume': ['mean', 'std'],
        'revenue': ['mean', 'sum']
    }).round(2)
    
    # Calculate correlations
    price_pivot = data.pivot(index='date', columns='route', values='price')
    price_corr = price_pivot.corr()
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Comparison', 'Volume Comparison',
                       'Revenue Comparison', 'Price Correlation')
    )
    
    # Add price comparison
    for route in data['route'].unique():
        route_data = data[data['route'] == route]
        fig.add_trace(
            go.Scatter(x=route_data['date'], y=route_data['price'],
                      name=f'{route} Price'),
            row=1, col=1
        )
    
    # Add volume comparison
    for route in data['route'].unique():
        route_data = data[data['route'] == route]
        fig.add_trace(
            go.Scatter(x=route_data['date'], y=route_data['volume'],
                      name=f'{route} Volume'),
            row=1, col=2
        )
    
    # Add revenue comparison
    for route in data['route'].unique():
        route_data = data[data['route'] == route]
        fig.add_trace(
            go.Scatter(x=route_data['date'], y=route_data['revenue'],
                      name=f'{route} Revenue'),
            row=2, col=1
        )
    
    # Add correlation heatmap
    fig.add_trace(
        go.Heatmap(z=price_corr.values,
                   x=price_corr.index,
                   y=price_corr.columns,
                   colorscale='RdBu'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=1000,
        title_text=f"Route Comparison - {size}",
        title_x=0.5
    )
    
    return fig, route_metrics, price_corr

# ---- Visualization Functions ----
def create_impact_plot(results, height=400):
    """Create visualization for price change impact."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Price Impact', 'Volume Impact', 'Revenue Impact')
    )
    
    # Add price bars
    fig.add_trace(
        go.Bar(x=['Base', 'New'], 
               y=[results['base_price'], results['new_price']],
               marker_color=['rgb(173,216,230)', 'rgb(0,0,139)']),
        row=1, col=1
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(x=['Base', 'New'], 
               y=[results['base_volume'], results['new_volume']],
               marker_color=['rgb(144,238,144)', 'rgb(0,100,0)']),
        row=1, col=2
    )
    
    # Add revenue bars
    fig.add_trace(
        go.Bar(x=['Base', 'New'], 
               y=[results['base_revenue'], results['new_revenue']],
               marker_color=['rgb(255,182,193)', 'rgb(139,0,0)']),
        row=1, col=3
    )
    
    fig.update_layout(
        height=height,
        showlegend=False,
        title_text=f"Impact Analysis (Elasticity: {results['elasticity']:.2f})",
        title_x=0.5
    )
    
    fig.update_yaxes(tickformat="$,.0f", title_text="Price", row=1, col=1)
    fig.update_yaxes(tickformat=",.0f", title_text="Volume", row=1, col=2)
    fig.update_yaxes(tickformat="$,.0f", title_text="Revenue", row=1, col=3)
    
    return fig

def create_demand_plot(results, prices, volumes, revenues, height=400):
    """Create visualization for demand and revenue curves."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Demand Curve', 'Revenue Curve')
    )
    
    # Add demand curve
    fig.add_trace(
        go.Scatter(x=prices, y=volumes, mode='lines', name='Demand'),
        row=1, col=1
    )
    
    # Add revenue curve
    fig.add_trace(
        go.Scatter(x=prices, y=revenues, mode='lines', name='Revenue'),
        row=1, col=2
    )
    
    # Add current points
    fig.add_trace(
        go.Scatter(x=[results['new_price']], y=[results['new_volume']],
                  mode='markers', marker=dict(size=10, color='red'),
                  name='Current Point'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=[results['new_price']], y=[results['new_revenue']],
                  mode='markers', marker=dict(size=10, color='red'),
                  name='Current Revenue'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=height,
        showlegend=True,
        title_text="Demand and Revenue Analysis",
        title_x=0.5
    )
    
    fig.update_xaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=1, col=2)
    
    return fig

# Nuove funzioni per l'analisi dei dati
def create_elasticity_analysis_plot(df, route, size):
    mask = (df['route'] == route) & (df['container_size'] == size)
    data = df[mask].copy()
    
    # Calcola variazioni percentuali
    data['price_pct_change'] = data['price'].pct_change()
    data['volume_pct_change'] = data['volume'].pct_change()
    
    # Calcola elasticità puntuale
    data['point_elasticity'] = data['volume_pct_change'] / data['price_pct_change']
    
    # Filtra valori validi per il plot
    valid_mask = (abs(data['price_pct_change']) > 0.001) & \
                 ~np.isinf(data['point_elasticity']) & \
                 (abs(data['point_elasticity']) < 10)
    plot_data = data[valid_mask]
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=('Price vs Volume Over Time',
                                     'Price vs Volume Scatter',
                                     'Point Elasticities Over Time',
                                     'Price Changes vs Volume Changes'))
    
    # Price and Volume over time
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['price'],
                  name='Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['volume'],
                  name='Volume', line=dict(color='green'),
                  yaxis='y2'),
        row=1, col=1
    )
    
    # Price vs Volume scatter
    fig.add_trace(
        go.Scatter(x=data['price'], y=data['volume'],
                  mode='markers', name='Price vs Volume'),
        row=1, col=2
    )
    
    # Point elasticities over time
    fig.add_trace(
        go.Scatter(x=plot_data['date'], y=plot_data['point_elasticity'],
                  mode='markers+lines', name='Point Elasticity'),
        row=2, col=1
    )
    
    # Price changes vs Volume changes
    fig.add_trace(
        go.Scatter(x=plot_data['price_pct_change'], 
                  y=plot_data['volume_pct_change'],
                  mode='markers', name='Changes Correlation'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Elasticity Analysis for {route} - {size}",
        title_x=0.5
    )
    
    # Add second y-axis for first plot
    fig.update_layout(yaxis2=dict(
        title="Volume",
        overlaying="y",
        side="right"
    ))
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Price", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Price % Change", row=2, col=2)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text="Point Elasticity", row=2, col=1)
    fig.update_yaxes(title_text="Volume % Change", row=2, col=2)
    
    return fig

def calculate_elasticity_statistics(df, route, size):
    mask = (df['route'] == route) & (df['container_size'] == size)
    data = df[mask].copy()
    
    data['price_pct_change'] = data['price'].pct_change()
    data['volume_pct_change'] = data['volume'].pct_change()
    data['point_elasticity'] = data['volume_pct_change'] / data['price_pct_change']
    
    valid_mask = (abs(data['price_pct_change']) > 0.001) & \
                 ~np.isinf(data['point_elasticity']) & \
                 (abs(data['point_elasticity']) < 10)
    
    elasticities = data.loc[valid_mask, 'point_elasticity']
    
    stats = {
        'median_elasticity': elasticities.median(),
        'mean_elasticity': elasticities.mean(),
        'std_elasticity': elasticities.std(),
        'min_elasticity': elasticities.min(),
        'max_elasticity': elasticities.max(),
        'observations': len(elasticities)
    }
    
    return stats


# ---- Main Application ----
def main():
    st.set_page_config(page_title="Dynamic Pricing Simulator", layout="wide")
    
    # Sidebar layout settings
    st.sidebar.header("Layout Settings")
    layout_container = st.sidebar.expander("Display Settings", expanded=False)
    with layout_container:
        chart_height = st.slider("Chart Height", 200, 800, 400, 50)
        content_width = st.slider("Content Width", 500, 2000, 1200, 100)
        metrics_cols = st.radio("Metrics Layout", 
                              ["2 columns", "3 columns", "4 columns"], 
                              index=2)
    
    n_cols = int(metrics_cols[0])
    
    # Main header
    st.title("Dynamic Pricing Simulator")
    st.markdown("""
    This simulator helps analyze the impact of price changes on shipping volumes and revenues.
    It uses historical data to estimate price elasticity and provide optimization suggestions.
    """)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = generate_sample_data()
        st.session_state.elasticities = calculate_elasticities(st.session_state.df)
    
    # Main controls
    st.sidebar.header("Controls")
    route = st.sidebar.selectbox("Select Route", 
                               sorted(st.session_state.df['route'].unique()))
    size = st.sidebar.selectbox("Select Container Size", 
                              sorted(st.session_state.df['container_size'].unique()))
    
    # Custom width container
    container = st.container()
    with container:
        st.markdown(f"""
            <style>
                .reportview-container .main .block-container{{
                    max-width: {content_width}px;
                    padding-top: 1rem;
                    padding-right: 1rem;
                    padding-left: 1rem;
                    padding-bottom: 1rem
                }}
            </style>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tab1, tab2, tab3, tab4,tab5 = st.tabs([
            "Price Simulation",
            "Optimization Analysis",
            "Data Analysis",
            "Advanced Analytics",
            "Elasticity Analysis",
        ])
        
        # Tab 1: Price Simulation
        with tab1:
            price_change = st.slider("Price Change %", -50, 50, 0)
            
            results = calculate_impact(
                st.session_state.df,
                route,
                size,
                price_change,
                st.session_state.elasticities[(route, size)]
            )
            
            # Display metrics
            cols = st.columns(n_cols)
            metrics = [
                ("Price", f"${results['new_price']:.0f}", 
                 f"{((results['new_price']/results['base_price'])-1)*100:.1f}%"),
                ("Volume", f"{results['new_volume']:.0f}", 
                 f"{((results['new_volume']/results['base_volume'])-1)*100:.1f}%"),
                ("Revenue", f"${results['new_revenue']:,.0f}", 
                 f"{results['revenue_change_pct']:.1f}%"),
                ("Elasticity", f"{results['elasticity']:.2f}", None)
            ]
            
            for i, (label, value, delta) in enumerate(metrics):
                with cols[i % n_cols]:
                    if delta:
                        st.metric(label, value, delta)
                    else:
                        st.metric(label, value)
            
            # Display impact plots
            st.plotly_chart(
                create_impact_plot(results, height=chart_height),
                use_container_width=True
            )
            
            prices, volumes, revenues = generate_demand_curve(
                results['base_price'],
                results['base_volume'],
                results['elasticity']
            )
            
            st.plotly_chart(
                create_demand_plot(results, prices, volumes, revenues, height=chart_height),
                use_container_width=True
            )
        
        # Tab 2: Optimization Analysis
        with tab2:
            test_elasticity = st.slider("Test Elasticity", -2.0, 1.0, -0.3, 0.01)
            
            base_data = st.session_state.df[
                (st.session_state.df['route'] == route) & 
                (st.session_state.df['container_size'] == size)
            ].iloc[-1]
            
            optimal_price = calculate_optimal_price(
                base_data['price'],
                base_data['volume'],
                test_elasticity
            )
            
            optimal_change = (optimal_price - base_data['price']) / base_data['price']
            optimal_volume = base_data['volume'] * (1 + test_elasticity * optimal_change)
            optimal_revenue = optimal_price * optimal_volume
            current_revenue = base_data['price'] * base_data['volume']
            
            st.subheader("Optimization Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Price", f"${optimal_price:.0f}", 
                         f"{((optimal_price/base_data['price'])-1)*100:.1f}%")
            with col2:
                st.metric("Projected Volume", f"{optimal_volume:.0f}", 
                         f"{((optimal_volume/base_data['volume'])-1)*100:.1f}%")
            with col3:
                st.metric("Projected Revenue", f"${optimal_revenue:,.0f}", 
                         f"{((optimal_revenue/current_revenue)-1)*100:.1f}%")
            
            optimization_results = {
                'base_price': base_data['price'],
                'new_price': optimal_price,
                'base_volume': base_data['volume'],
                'new_volume': optimal_volume,
                'base_revenue': current_revenue,
                'new_revenue': optimal_revenue,
                'elasticity': test_elasticity
            }
            
            prices, volumes, revenues = generate_demand_curve(
                base_data['price'],
                base_data['volume'],
                test_elasticity
            )
            
            st.plotly_chart(
                create_demand_plot(optimization_results, prices, volumes, revenues, 
                                 height=chart_height),
                use_container_width=True
            )
        
        # Tab 3: Data Analysis
        with tab3:
            st.subheader("Historical Data Analysis")
            mask = (st.session_state.df['route'] == route) & \
                   (st.session_state.df['container_size'] == size)
            
            fig = make_subplots(rows=3, cols=1,
                              subplot_titles=('Price Trend', 'Volume Trend', 'Revenue Trend'))
            
            data = st.session_state.df[mask].sort_values('date')
            
            # Price trend
            fig.add_trace(
                go.Scatter(x=data['date'], y=data['price'],
                          name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Volume trend
            fig.add_trace(
                go.Scatter(x=data['date'], y=data['volume'],
                          name='Volume', line=dict(color='green')),
                row=2, col=1
            )
            
            # Revenue trend
            fig.add_trace(
                go.Scatter(x=data['date'], y=data['revenue'],
                          name='Revenue', line=dict(color='red')),
                row=3, col=1
            )
            
            fig.update_layout(height=900, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data
            if st.checkbox("Show Raw Data"):
                st.dataframe(data)
        
        # Tab 4: Advanced Analytics
        with tab4:
            st.header("Advanced Analytics")
            
            # Seasonal Analysis
            st.subheader("Seasonal Patterns")
            seasonal_fig, seasonal_stats = analyze_seasonality(
                st.session_state.df, route, size
            )
            st.plotly_chart(seasonal_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Price Seasonality")
                st.write(f"Peak Month: {seasonal_stats['price_seasonality']['peak_month']}")
                st.write(f"Trough Month: {seasonal_stats['price_seasonality']['trough_month']}")
                st.write(f"Amplitude: ${seasonal_stats['price_seasonality']['seasonal_amplitude']:.2f}")
            
            with col2:
                st.markdown("#### Volume Seasonality")
                st.write(f"Peak Month: {seasonal_stats['volume_seasonality']['peak_month']}")
                st.write(f"Trough Month: {seasonal_stats['volume_seasonality']['trough_month']}")
                st.write(f"Amplitude: {seasonal_stats['volume_seasonality']['seasonal_amplitude']:.0f} units")
            
            # Future Forecasts
            st.subheader("Future Forecasts")
            forecast_fig, forecast_data = forecast_future_values(
                st.session_state.df, route, size
            )
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            if st.checkbox("Show Forecast Data"):
                st.dataframe(forecast_data)
            
            # Route Comparison
            st.subheader("Route Comparison")
            comparison_fig, route_metrics, price_corr = compare_routes(
                st.session_state.df, size
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            if st.checkbox("Show Route Metrics"):
                st.dataframe(route_metrics)
            
            if st.checkbox("Show Price Correlations"):
                st.dataframe(price_corr)

        with tab5:
            st.subheader("Data and Elasticity Analysis")
            
            # Mostra statistiche di elasticità
            stats = calculate_elasticity_statistics(
                st.session_state.df,
                route,
                size
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Median Elasticity", f"{stats['median_elasticity']:.3f}")
                st.metric("Mean Elasticity", f"{stats['mean_elasticity']:.3f}")
            with col2:
                st.metric("Std Deviation", f"{stats['std_elasticity']:.3f}")
                st.metric("Observations", f"{stats['observations']}")
            with col3:
                st.metric("Min Elasticity", f"{stats['min_elasticity']:.3f}")
            with col4:
                st.metric("Max Elasticity", f"{stats['max_elasticity']:.3f}")
            
            # Aggiungi interpretazione
            st.markdown("### Interpretation")
            
            median_el = stats['median_elasticity']
            if median_el > -1:
                elasticity_type = "inelastic"
                revenue_impact = "increase"
            else:
                elasticity_type = "elastic"
                revenue_impact = "decrease"
            
            st.markdown(f"""
            #### Key Findings:
            - The demand for this route-size combination is **{elasticity_type}** (median elasticity: {median_el:.2f})
            - This means that a 1% increase in price leads to a {abs(median_el):.2f}% {revenue_impact} in demand
            - With {stats['observations']} observations, the elasticity estimates range from {stats['min_elasticity']:.2f} to {stats['max_elasticity']:.2f}
            
            #### Implications:
            - {"Price increases will likely lead to revenue gains" if median_el > -1 else "Price increases will likely lead to revenue losses"}
            - {"Consider strategic price increases" if median_el > -1 else "Consider maintaining or reducing prices"}
            - The standard deviation of {stats['std_elasticity']:.2f} suggests {"high" if stats['std_elasticity'] > 1 else "moderate" if stats['std_elasticity'] > 0.5 else "low"} variability in customer responses to price changes
            """)
            
            # Mostra grafici di analisi
            st.plotly_chart(
                create_elasticity_analysis_plot(st.session_state.df, route, size),
                use_container_width=True
            )
            
            # Mostra dati grezzi
            st.subheader("Raw Data")
            mask = (st.session_state.df['route'] == route) & \
                   (st.session_state.df['container_size'] == size)
            st.dataframe(st.session_state.df[mask])


if __name__ == "__main__":
    main()