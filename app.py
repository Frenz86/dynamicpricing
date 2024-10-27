import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generazione dati di esempio
def generate_sample_data(n_weeks=52):
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
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
                price = base_price * seasonal_factor * (1 + np.random.uniform(-0.2, 0.2))
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

# Calcolo elasticità
def calculate_elasticities(df):
    elasticities = {}
    for route in df['route'].unique():
        for size in df['container_size'].unique():
            mask = (df['route'] == route) & (df['container_size'] == size)
            route_data = df[mask].copy()
            
            price_pct_change = route_data['price'].pct_change()
            volume_pct_change = route_data['volume'].pct_change()
            
            valid_mask = (abs(price_pct_change) > 0.001) & ~np.isinf(volume_pct_change/price_pct_change)
            if valid_mask.sum() > 0:
                point_elasticities = volume_pct_change[valid_mask] / price_pct_change[valid_mask]
                point_elasticities = point_elasticities[abs(point_elasticities) < 10]
                elasticity = point_elasticities.median()
            else:
                elasticity = -0.3
            
            elasticity = np.clip(elasticity, -2.0, -0.1)
            elasticities[(route, size)] = round(elasticity, 3)
    
    return elasticities

# Calcolo impatto variazioni prezzo
def calculate_impact(df, route, size, price_change_pct, elasticity):
    mask = (df['route'] == route) & (df['container_size'] == size)
    base_data = df[mask].iloc[-1]
    
    base_price = base_data['price']
    base_volume = base_data['volume']
    
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

# Calcolo prezzo ottimale
def calculate_optimal_price(base_price, base_volume, elasticity):
    if elasticity >= -1:
        return base_price * 1.5
    else:
        optimal_markup = -1 / elasticity
        return base_price * (1 + optimal_markup)

# Generazione curva di domanda
def generate_demand_curve(base_price, base_volume, elasticity, points=50):
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

# Creazione grafici impatto
def create_impact_plot(results, height=400):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Price Impact', 'Volume Impact', 'Revenue Impact')
    )
    
    fig.add_trace(
        go.Bar(x=['Base', 'New'], 
               y=[results['base_price'], results['new_price']],
               marker_color=['rgb(173,216,230)', 'rgb(0,0,139)']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=['Base', 'New'], 
               y=[results['base_volume'], results['new_volume']],
               marker_color=['rgb(144,238,144)', 'rgb(0,100,0)']),
        row=1, col=2
    )
    
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
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Demand Curve', 'Revenue Curve')
    )
    
    fig.add_trace(
        go.Scatter(x=prices, y=volumes, mode='lines', name='Demand'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=prices, y=revenues, mode='lines', name='Revenue'),
        row=1, col=2
    )
    
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

def main():
    st.set_page_config(page_title="Dynamic Pricing Simulator", layout="wide")
    
    # Layout settings
    st.sidebar.header("Layout Settings")
    layout_container = st.sidebar.expander("Display Settings", expanded=False)
    with layout_container:
        chart_height = st.slider("Chart Height", 200, 800, 400, 50)
        content_width = st.slider("Content Width", 500, 2000, 1200, 100)
        metrics_cols = st.radio("Metrics Layout", ["2 columns", "3 columns", "4 columns"], index=2)
    
    n_cols = int(metrics_cols[0])
    
    # Header
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
    route = st.sidebar.selectbox("Select Route", sorted(st.session_state.df['route'].unique()))
    size = st.sidebar.selectbox("Select Container Size", sorted(st.session_state.df['container_size'].unique()))
    
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
        
        tab1, tab2, tab3 = st.tabs(["Price Simulation", "Optimization Analysis", "Data Analysis"])
        
        with tab1:
            price_change = st.slider("Price Change %", -50, 50, 0)
            
            results = calculate_impact(
                st.session_state.df,
                route,
                size,
                price_change,
                st.session_state.elasticities[(route, size)]
            )
            
            cols = st.columns(n_cols)
            metrics = [
                ("Price", f"${results['new_price']:.2f}", 
                 f"{((results['new_price']/results['base_price'])-1)*100:.1f}%"),
                ("Volume", f"{results['new_volume']:.0f}", 
                 f"{((results['new_volume']/results['base_volume'])-1)*100:.1f}%"),
                ("Revenue", f"${results['new_revenue']:,.0f}", 
                 f"{results['revenue_change_pct']:.1f}%"),
                ("Elasticity", f"{results['elasticity']:.3f}", None)
            ]
            
            for i, (label, value, delta) in enumerate(metrics):
                with cols[i % n_cols]:
                    if delta:
                        st.metric(label, value, delta)
                    else:
                        st.metric(label, value)
            
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
        
        with tab2:
            test_elasticity = st.slider("Test Elasticity", -2.0, -0.1, -0.3, 0.1)
            
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
                st.metric("Optimal Price", f"${optimal_price:.2f}", 
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
                create_demand_plot(optimization_results, prices, volumes, revenues, height=chart_height),
                use_container_width=True
            )
            
        with tab3:
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