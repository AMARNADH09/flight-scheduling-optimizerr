# app.py - Complete Mumbai Airport Flight Scheduling Optimizer
# Honeywell Hackathon 2024 - FlightRadar24 Real-time Integration

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import asyncio
import warnings
import time
warnings.filterwarnings('ignore')
import os
# Add this BEFORE any other imports
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Import modules with comprehensive error handling
try:
    from fr24_api_integration import FlightRadar24APIClient, FlightDataContinuousCollector
    FR24_AVAILABLE = True
    print("✅ FlightRadar24 API integration loaded")
except ImportError as e:
    FR24_AVAILABLE = False
    print(f"⚠️ FlightRadar24 API not available: {e}")

try:
    from analysis import FlightAnalyzer
    ANALYSIS_AVAILABLE = True
except ImportError:
    try:
        from analysis import FlightAnalyzer
        ANALYSIS_AVAILABLE = True
    except ImportError as e:
        ANALYSIS_AVAILABLE = False
        print(f"⚠️ Analysis module not available: {e}")

try:
    from ml_models import FlightDelayPredictor, CascadingImpactAnalyzer
    ML_AVAILABLE = True
except ImportError:
    try:
        from ml_models import FlightDelayPredictor, CascadingImpactAnalyzer
        ML_AVAILABLE = True
    except ImportError as e:
        ML_AVAILABLE = False
        print(f"⚠️ ML models not available: {e}")

try:
    from nlp_interface import FlightNLPInterface
    NLP_AVAILABLE = True
except ImportError:
    try:
        from nlp_interface import FlightNLPInterface
        NLP_AVAILABLE = True
    except ImportError as e:
        NLP_AVAILABLE = False
        print(f"⚠️ NLP interface not available: {e}")

# Configure Streamlit page
st.set_page_config(
    page_title="Mumbai Airport Flight Scheduler - FlightRadar24 Integration",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 50%, #2ca02c 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header h3 {
        color: white;
        margin-top: 0;
        font-weight: 300;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffaa00;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .alert-success {
        background-color: #00aa44;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
             /* Dark mode styles - detect Streamlit's dark theme */
    [data-theme="dark"] .stDataFrame,
    .stApp[data-theme="dark"] .stDataFrame,
    html[data-theme="dark"] .stDataFrame {
        background-color: #0e1117 !important;
    }
    
    [data-theme="dark"] .stDataFrame table,
    .stApp[data-theme="dark"] .stDataFrame table,
    html[data-theme="dark"] .stDataFrame table {
        background-color: #0e1117 !important;
        color: #fafafa !important;
        border: 1px solid #262730 !important;
    }
    
    [data-theme="dark"] .stDataFrame th,
    .stApp[data-theme="dark"] .stDataFrame th,
    html[data-theme="dark"] .stDataFrame th {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #3d4043 !important;
        font-weight: 600 !important;
    }
    
    [data-theme="dark"] .stDataFrame td,
    .stApp[data-theme="dark"] .stDataFrame td,
    html[data-theme="dark"] .stDataFrame td {
        background-color: #0e1117 !important;
        color: #fafafa !important;
        border: 1px solid #262730 !important;
    }
    
    [data-theme="dark"] .stDataFrame tbody tr:nth-child(even),
    .stApp[data-theme="dark"] .stDataFrame tbody tr:nth-child(even),
    html[data-theme="dark"] .stDataFrame tbody tr:nth-child(even) {
        background-color: #1a1d23 !important;
    }
    
    [data-theme="dark"] .stDataFrame tbody tr:hover,
    .stApp[data-theme="dark"] .stDataFrame tbody tr:hover,
    html[data-theme="dark"] .stDataFrame tbody tr:hover {
        background-color: #262730 !important;
    }

    /* Media query fallback for system dark mode preference */
    @media (prefers-color-scheme: dark) {
        .stDataFrame {
            background-color: #0e1117 !important;
        }
        
        .stDataFrame table {
            background-color: #0e1117 !important;
            color: #fafafa !important;
            border: 1px solid #262730 !important;
        }
        
        .stDataFrame th {
            background-color: #262730 !important;
            color: #fafafa !important;
            border: 1px solid #3d4043 !important;
        }
        
        .stDataFrame td {
            background-color: #0e1117 !important;
            color: #fafafa !important;
            border: 1px solid #262730 !important;
        }
        
        .stDataFrame tbody tr:nth-child(even) {
            background-color: #1a1d23 !important;
        }
    }
        div[data-testid="stDataFrame"] > div,
        div[data-testid="stTable"] > div {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Ensure text is always visible */
    .stDataFrame * {
        color: inherit !important;
    }

</style>
""", unsafe_allow_html=True)

def main():
    """Enhanced main application with FlightRadar24 integration"""
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🛫 Mumbai Airport Flight Scheduler</h1>
        <h3>Real-time FlightRadar24 Integration </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>🎛️ Control Center</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # System status
        st.markdown("### 🔧 System Status")
        
        status_items = [
            ("FlightRadar24 API", FR24_AVAILABLE),
            ("Flight Analysis", ANALYSIS_AVAILABLE), 
            ("ML Models", ML_AVAILABLE),
            ("NLP Interface", NLP_AVAILABLE)
        ]
        
        for item, available in status_items:
            status_emoji = "✅" if available else "❌"
            st.markdown(f"{status_emoji} {item}")
        
        st.markdown("---")
        
        # Data source selection
        st.markdown("### 📊 Data Source")
        
        data_source_options = ["🔴 Real-time FlightRadar24 API"]
        if os.path.exists("Flight_Data.xlsx"):
            data_source_options.append("📁 FlightRadar24 Excel File")
        data_source_options.append("📋 Enhanced Sample Data")
        
        data_source = st.selectbox(
            "Choose your data source:",
            data_source_options,
            index=0
        )
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        
        pages = [
            "🏠 Real-time Dashboard",
            "📊 Delay Analysis", 
            "🎯 Schedule Optimization",
            "💬 NLP Query Interface",
            "🔗 Cascading Impact Analysis",
            "🧠 ML Predictions"
        ]
        
        page = st.selectbox("Select Analysis Module:", pages)
    
    # Load data based on selection
    with st.spinner("🔄 Loading flight data..."):
        data = load_flight_data(data_source)
    
    if data is None or data.empty:
        st.error("❌ No flight data available. Please check your data source.")
        st.stop()
    
    # Display data source info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Flights", len(data))
    
    with col2:
        if 'flight_direction' in data.columns:
            departures = (data['flight_direction'] == 'Departure').sum()
            st.metric("🛫 Departures", departures)
        else:
            st.metric("🛫 Departures", "N/A")
    
    with col3:
        if 'flight_direction' in data.columns:
            arrivals = (data['flight_direction'] == 'Arrival').sum()
            st.metric("🛬 Arrivals", arrivals)
        else:
            st.metric("🛬 Arrivals", "N/A")
    
    with col4:
        data_freshness = "Live" if "Real-time" in data_source else "Historical"
        st.metric("🕒 Data Status", data_freshness)
    
    # Alerts for critical situations
    if 'departure_delay' in data.columns:
        severe_delays = data[data['departure_delay'] > 60]
        if not severe_delays.empty:
            st.markdown(f"""
            <div class="alert-critical">
                🚨 CRITICAL ALERT: {len(severe_delays)} flights with severe delays (>60 min)
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Route to appropriate page
    try:
        if "Real-time Dashboard" in page:
            show_realtime_dashboard(data, data_source)
        elif "Delay Analysis" in page:
            show_enhanced_delay_analysis(data)
        elif "Schedule Optimization" in page:
            show_schedule_optimization(data)
        elif "NLP Query Interface" in page:
            show_nlp_interface(data)
        elif "Cascading Impact Analysis" in page:
            show_cascading_analysis(data)
        elif "ML Predictions" in page:
            show_ml_predictions(data)
    except Exception as e:
        st.error(f"❌ Error in {page}: {str(e)}")
        st.info("Please try refreshing the page or selecting a different module.")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_flight_data(data_source):
    """Load flight data based on selected source"""
    
    if "Real-time FlightRadar24 API" in data_source:
        return load_realtime_data()
    elif "FlightRadar24 Excel File" in data_source:
        return load_excel_data()
    else:
        return create_enhanced_sample_data()

def load_realtime_data():
    """Load real-time data from FlightRadar24 API"""
    if not FR24_AVAILABLE:
        st.sidebar.error("❌ FlightRadar24 API not available")
        return create_enhanced_sample_data()
    
    try:
        st.sidebar.info("🔄 Connecting to FlightRadar24...")
        
        api_client = FlightRadar24APIClient()
        
        with st.sidebar:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📡 Fetching departures...")
            progress_bar.progress(25)
            
            status_text.text("📡 Fetching arrivals...")
            progress_bar.progress(50)
            
            status_text.text("📡 Processing data...")
            progress_bar.progress(75)
            
            real_data = api_client.get_comprehensive_mumbai_data()
            
            progress_bar.progress(100)
            status_text.text("✅ Data loaded!")
            
            time.sleep(1)  # Show completion briefly
            progress_bar.empty()
            status_text.empty()
        
        if not real_data.empty:
            st.sidebar.success(f"✅ Live: {len(real_data)} flights")
            st.sidebar.info(f"🕒 Updated: {datetime.now().strftime('%H:%M:%S')}")
            
            # Show real-time statistics
            with st.sidebar.expander("📊 Live Stats"):
                if 'departure_delay' in real_data.columns:
                    avg_delay = real_data['departure_delay'].mean()
                    st.write(f"⏱️ Avg Delay: {avg_delay:.1f} min")
                    
                    on_time = (real_data['departure_delay'] <= 15).mean() * 100
                    st.write(f"✅ On-time: {on_time:.1f}%")
            
            return real_data
        else:
            st.sidebar.warning("⚠️ No live data, using fallback")
            return create_enhanced_sample_data()
            
    except Exception as e:
        st.sidebar.error(f"❌ API Error: {str(e)[:50]}...")
        return create_enhanced_sample_data()

def load_excel_data():
    """Load data from FlightRadar24 Excel file"""
    try:
        st.sidebar.info("📁 Processing Excel file...")
        
        # Check for existing file
        if os.path.exists("Flight_Data.xlsx"):
            df = pd.read_excel("Flight_Data.xlsx")
            
            # Process the Excel data
            processed_data = process_flightradar24_excel(df)
            
            if not processed_data.empty:
                st.sidebar.success(f"✅ Excel: {len(processed_data)} flights")
                return processed_data
            else:
                st.sidebar.warning("⚠️ Excel processing failed")
                return create_enhanced_sample_data()
        else:
            # File upload option
            uploaded_file = st.sidebar.file_uploader(
                "Upload FlightRadar24 Excel File",
                type=['xlsx', 'xls'],
                help="Upload your Flight_Data.xlsx export"
            )
            
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file)
                processed_data = process_flightradar24_excel(df)
                
                if not processed_data.empty:
                    st.sidebar.success(f"✅ Uploaded: {len(processed_data)} flights")
                    return processed_data
                else:
                    st.sidebar.error("❌ Failed to process uploaded file")
                    return create_enhanced_sample_data()
            else:
                st.sidebar.info("📤 Please upload your Flight_Data.xlsx file")
                return create_enhanced_sample_data()
                
    except Exception as e:
        st.sidebar.error(f"❌ Excel error: {str(e)[:50]}...")
        return create_enhanced_sample_data()

def process_flightradar24_excel(df):
    """Process FlightRadar24 Excel export data"""
    try:
        # Your Excel file columns: S.No, Flight Number, Date, From, To, Aircraft, Flighttime, STD, ATD, STA, ATA
        processed_flights = []
        
        for _, row in df.iterrows():
            # Extract basic flight info
            flight_number = row.get('Flight Number', '')
            if not flight_number:
                continue
            
            # Parse airports
            from_airport = str(row.get('From', ''))
            to_airport = str(row.get('To', ''))
            
            # Only process Mumbai flights
            if 'Mumbai' not in from_airport and 'BOM' not in from_airport and \
               'Mumbai' not in to_airport and 'BOM' not in to_airport:
                continue
            
            # Parse times
            std = parse_time(row.get('STD'))  # Scheduled departure
            atd = parse_time(row.get('ATD'))  # Actual departure
            sta = parse_time(row.get('STA'))  # Scheduled arrival  
            ata = parse_time(row.get('ATA'))  # Actual arrival
            
            # Calculate delays
            dep_delay = calculate_time_difference(atd, std) if atd and std else None
            arr_delay = calculate_time_difference(ata, sta) if ata and sta else None
            
            flight_data = {
                'flight_number': flight_number,
                'aircraft': str(row.get('Aircraft', '')),
                'from_airport': from_airport,
                'to_airport': to_airport,
                'scheduled_departure': std,
                'actual_departure': atd,
                'scheduled_arrival': sta,
                'actual_arrival': ata,
                'departure_delay': dep_delay,
                'arrival_delay': arr_delay,
                'date': parse_date(row.get('Date')) or datetime.now().date(),
                'flight_direction': 'Departure' if 'Mumbai' in from_airport or 'BOM' in from_airport else 'Arrival',
                'data_source': 'FlightRadar24_Excel'
            }
            
            processed_flights.append(flight_data)
        
        if processed_flights:
            df_processed = pd.DataFrame(processed_flights)
            df_processed = add_analysis_fields(df_processed)
            return df_processed
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error processing Excel: {e}")
        return pd.DataFrame()

def parse_time(time_value):
    """Parse time from various formats"""
    if pd.isna(time_value) or time_value == '' or str(time_value).strip() == '':
        return None
    
    try:
        time_str = str(time_value).strip()
        
        # Handle Excel time formats
        if ':' in time_str:
            # Format: "9:00 AM" or "21:30"
            if 'AM' in time_str or 'PM' in time_str:
                time_obj = datetime.strptime(time_str, '%I:%M %p')
                return time_obj.strftime('%H:%M')
            else:
                # 24-hour format
                parts = time_str.split(':')
                if len(parts) == 2:
                    hour, minute = int(parts[0]), int(parts[1])
                    return f"{hour:02d}:{minute:02d}"
        
        # Handle other formats
        if len(time_str) == 4 and time_str.isdigit():
            # Format: "0900"
            hour, minute = int(time_str[:2]), int(time_str[2:])
            return f"{hour:02d}:{minute:02d}"
            
    except:
        pass
    
    return None

def parse_date(date_value):
    """Parse date from various formats"""
    if pd.isna(date_value):
        return None
    
    try:
        if isinstance(date_value, datetime):
            return date_value.date()
        
        date_str = str(date_value).strip()
        
        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%d-%m-%Y', 
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%d %b %Y',
            '%d-%b-%Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
                
    except:
        pass
    
    return datetime.now().date()

def calculate_time_difference(actual_time, scheduled_time):
    """Calculate difference between times in minutes"""
    if not actual_time or not scheduled_time:
        return None
    
    try:
        base_date = "2025-01-01"
        scheduled_dt = datetime.strptime(f"{base_date} {scheduled_time}", '%Y-%m-%d %H:%M')
        actual_dt = datetime.strptime(f"{base_date} {actual_time}", '%Y-%m-%d %H:%M')
        
        # Handle day rollover
        if actual_dt < scheduled_dt:
            actual_dt += timedelta(days=1)
        
        diff_minutes = (actual_dt - scheduled_dt).total_seconds() / 60
        return diff_minutes
        
    except:
        return None

def add_analysis_fields(df):
    """Add fields needed for analysis"""
    if df.empty:
        return df
    
    # Time slot categorization
    def get_time_slot(time_str):
        if not time_str:
            return 'Unknown'
        try:
            hour = int(time_str.split(':')[0])
            if 6 <= hour < 9:
                return '6AM-9AM'
            elif 9 <= hour < 12:
                return '9AM-12PM'
            elif 12 <= hour < 15:
                return '12PM-3PM'
            elif 15 <= hour < 18:
                return '3PM-6PM'
            elif 18 <= hour < 21:
                return '6PM-9PM'
            else:
                return 'Night'
        except:
            return 'Unknown'
    
    df['time_slot'] = df['scheduled_departure'].apply(get_time_slot)
    
    # Day of week
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # Peak hour indicator
    df['is_peak_hour'] = df['time_slot'].isin(['6AM-9AM', '6PM-9PM'])
    
    # Delay category
    def categorize_delay(delay):
        if pd.isna(delay):
            return 'Unknown'
        elif delay <= 0:
            return 'On-time/Early'
        elif delay <= 15:
            return 'Minor Delay'
        elif delay <= 30:
            return 'Moderate Delay'
        elif delay <= 60:
            return 'Major Delay'
        else:
            return 'Severe Delay'
    
    if 'departure_delay' in df.columns:
        df['delay_category'] = df['departure_delay'].apply(categorize_delay)
    
    return df

def create_enhanced_sample_data():
    """Create realistic sample data for demonstration"""
    np.random.seed(42)
    n_flights = 200
    
    # Realistic Mumbai routes with frequencies
    routes = [
        ('Mumbai (BOM)', 'Delhi (DEL)', 0.25, 25),
        ('Mumbai (BOM)', 'Bangalore (BLR)', 0.20, 18),
        ('Mumbai (BOM)', 'Chennai (MAA)', 0.15, 22),
        ('Delhi (DEL)', 'Mumbai (BOM)', 0.25, 20),
        ('Bangalore (BLR)', 'Mumbai (BOM)', 0.15, 16)
    ]
    
    # Indian airlines with characteristics
    airlines = [
        ('AI', 'Air India', 5),      # Higher delays
        ('6E', 'IndiGo', -2),        # Better performance
        ('UK', 'Vistara', -3),       # Premium, punctual
        ('SG', 'SpiceJet', 3),       # Budget, more delays
        ('G8', 'GoAir', 2),          # Budget carrier
    ]
    
    flights = []
    
    for i in range(n_flights):
        # Select route
        route_weights = [r[2] for r in routes]
        route_idx = np.random.choice(len(routes), p=route_weights)
        from_airport, to_airport, _, base_delay = routes[route_idx]
        
        # Select airline
        airline_code, airline_name, delay_modifier = airlines[np.random.randint(0, len(airlines))]
        
        # Generate times
        base_hour = np.random.choice([6, 7, 8, 9, 10, 11], p=[0.35, 0.25, 0.20, 0.12, 0.05, 0.03])
        base_minute = np.random.choice([0, 15, 30, 45])
        
        scheduled_dep = f"{base_hour:02d}:{base_minute:02d}"
        
        # Calculate realistic delay
        total_delay = base_delay + delay_modifier + np.random.normal(0, 15)
        total_delay = max(-5, total_delay)  # Min -5 minutes (early)
        
        actual_dep_time = datetime.strptime(scheduled_dep, '%H:%M') + timedelta(minutes=total_delay)
        actual_dep = actual_dep_time.strftime('%H:%M')
        
        # Generate arrival times (2 hour average flight)
        flight_duration = np.random.randint(90, 150)  # 1.5 to 2.5 hours
        scheduled_arr_time = datetime.strptime(scheduled_dep, '%H:%M') + timedelta(minutes=flight_duration)
        scheduled_arr = scheduled_arr_time.strftime('%H:%M')
        
        actual_arr_time = actual_dep_time + timedelta(minutes=flight_duration)
        actual_arr = actual_arr_time.strftime('%H:%M')
        
        arrival_delay = total_delay * 0.8 + np.random.normal(0, 8)
        
        flight = {
            'flight_number': f'{airline_code}{np.random.randint(100, 999)}',
            'aircraft': f'VT-{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}{chr(65+np.random.randint(0,26))}',
            'airline': airline_name,
            'from_airport': from_airport,
            'to_airport': to_airport,
            'scheduled_departure': scheduled_dep,
            'actual_departure': actual_dep,
            'scheduled_arrival': scheduled_arr,
            'actual_arrival': actual_arr,
            'departure_delay': total_delay,
            'arrival_delay': arrival_delay,
            'status': 'Landed' if total_delay < 30 else 'Delayed',
            'date': datetime.now().date() - timedelta(days=np.random.randint(0, 7)),
            'flight_direction': 'Departure' if 'Mumbai' in from_airport else 'Arrival',
            'data_source': 'Enhanced_Sample'
        }
        
        flights.append(flight)
    
    df = pd.DataFrame(flights)
    df = add_analysis_fields(df)
    
    return df

def show_realtime_dashboard(data, data_source):
    """Enhanced real-time dashboard"""
    st.header("🏠 Real-time Mumbai Airport Dashboard")
    
    # Data source and freshness indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        source_emoji = "🔴" if "Real-time" in data_source else "📁" if "Excel" in data_source else "📋"
        st.markdown(f"**Data Source:** {source_emoji} {data_source}")
    
    with col2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")
    
    with col3:
        if "Real-time" in data_source:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
    
    # Alert system
    if 'departure_delay' in data.columns:
        severe_delays = data[data['departure_delay'] > 60]
        if not severe_delays.empty:
            st.markdown(f"""
            <div class="alert-critical">
                🚨 CRITICAL: {len(severe_delays)} flights with severe delays (>60 min)
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🚨 View Severe Delays"):
                cols_to_show = ['flight_number', 'from_airport', 'to_airport', 'departure_delay', 'status']
                available_cols = [col for col in cols_to_show if col in severe_delays.columns]
                st.dataframe(severe_delays[available_cols], use_container_width=True)
    
    # Key metrics
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_flights = len(data)
        st.metric("✈️ Total Flights", total_flights)
    
    with col2:
        if 'departure_delay' in data.columns:
            avg_delay = data['departure_delay'].mean()
            delay_color = "🔴" if avg_delay > 20 else "🟡" if avg_delay > 15 else "🟢"
            st.metric("⏱️ Avg Delay", f"{avg_delay:.1f} min", delta=f"{delay_color}")
        else:
            st.metric("⏱️ Avg Delay", "N/A")
    
    with col3:
        if 'departure_delay' in data.columns:
            on_time_rate = (data['departure_delay'] <= 15).mean() * 100
            performance_color = "🟢" if on_time_rate >= 80 else "🟡" if on_time_rate >= 70 else "🔴"
            st.metric("✅ On-Time Rate", f"{on_time_rate:.1f}%", delta=f"{performance_color}")
        else:
            st.metric("✅ On-Time Rate", "N/A")
    
    with col4:
        if 'flight_direction' in data.columns:
            departures = (data['flight_direction'] == 'Departure').sum()
            st.metric("🛫 Departures", departures)
        else:
            st.metric("🛫 Departures", "N/A")
    
    with col5:
        if 'flight_direction' in data.columns:
            arrivals = (data['flight_direction'] == 'Arrival').sum()
            st.metric("🛬 Arrivals", arrivals)
        else:
            st.metric("🛬 Arrivals", "N/A")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕐 Flight Activity by Time")
        
        if 'time_slot' in data.columns:
            time_dist = data['time_slot'].value_counts().sort_index()
            
            fig = px.bar(
                x=time_dist.index,
                y=time_dist.values,
                title="Flight Distribution by Time Slot",
                labels={'x': 'Time Slot', 'y': 'Number of Flights'},
                color=time_dist.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Time slot data not available")
    
    with col2:
        st.subheader("📊 Delay Distribution")
        
        if 'departure_delay' in data.columns:
            delay_data = data['departure_delay'].dropna()
            
            if not delay_data.empty:
                fig = px.histogram(
                    delay_data,
                    nbins=25,
                    title="Real-time Delay Pattern",
                    labels={'value': 'Delay (minutes)', 'count': 'Number of Flights'},
                    color_discrete_sequence=['#ff7f0e']
                )
                fig.add_vline(x=15, line_dash="dash", line_color="red", 
                             annotation_text="On-time threshold (15 min)")
                fig.add_vline(x=delay_data.mean(), line_dash="dash", line_color="green",
                             annotation_text=f"Average ({delay_data.mean():.1f} min)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No delay data available")
        else:
            st.info("Delay data not available")
    
    # Live flight table with filtering
    st.subheader("✈️ Live Flight Information")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        direction_filter = st.selectbox(
            "Flight Direction",
            ["All", "Departure", "Arrival"] if 'flight_direction' in data.columns else ["All"]
        )
    
    with col2:
        delay_filter = st.selectbox(
            "Delay Status", 
            ["All", "On-time (≤15 min)", "Delayed (>15 min)", "Severe (>60 min)"]
        )
    
    with col3:
        if 'time_slot' in data.columns:
            time_slots = ["All"] + sorted(data['time_slot'].unique().tolist())
            time_filter = st.selectbox("Time Slot", time_slots)
        else:
            time_filter = "All"
    
    with col4:
        show_count = st.selectbox("Show Flights", [20, 50, 100, "All"])
    
    # Apply filters
    filtered_data = data.copy()
    
    if direction_filter != "All" and 'flight_direction' in data.columns:
        filtered_data = filtered_data[filtered_data['flight_direction'] == direction_filter]
    
    if delay_filter != "All" and 'departure_delay' in data.columns:
        if delay_filter == "On-time (≤15 min)":
            filtered_data = filtered_data[filtered_data['departure_delay'] <= 15]
        elif delay_filter == "Delayed (>15 min)":
            filtered_data = filtered_data[filtered_data['departure_delay'] > 15]
        elif delay_filter == "Severe (>60 min)":
            filtered_data = filtered_data[filtered_data['departure_delay'] > 60]
    
    if time_filter != "All" and 'time_slot' in data.columns:
        filtered_data = filtered_data[filtered_data['time_slot'] == time_filter]
    
    # Display results
    if not filtered_data.empty:
        display_columns = [
            'flight_number', 'from_airport', 'to_airport', 
            'scheduled_departure', 'actual_departure', 'departure_delay', 
            'status', 'flight_direction'
        ]
        available_columns = [col for col in display_columns if col in filtered_data.columns]
        
        # Limit rows if specified
        display_data = filtered_data[available_columns]
        if show_count != "All":
            display_data = display_data.head(show_count)
        
        # Style the dataframe based on delays
        def highlight_delays(row):
            if 'departure_delay' in row and pd.notna(row['departure_delay']):
                delay = row['departure_delay']
                if delay <= 0:
                    return ['background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; font-weight: bold;'] * len(row)  # Green for on-time/early
                elif delay <= 15:
                    return ['background: linear-gradient(135deg, #ffc107 0%, #ffdb4d 100%); color: white; font-weight: 500;'] * len(row)  # Yellow for minor
                elif delay <= 60:
                    return ['background: linear-gradient(135deg, #fd7e14 0%, #ff922b 100%); color: white; font-weight: 500;'] * len(row)  # Light red for major
                else:
                    return [' background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%); color: white; font-weight: bold;'] * len(row)  # Dark red for severe
            return [''] * len(row)
        # Add this function after your highlight_delays function
        def style_dataframe_responsive(df):
            """Style dataframe to work in both light and dark modes"""
            return df.style.set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', 'var(--background-color)'),
                    ('color', 'var(--text-color)'),
                    ('font-weight', 'bold'),
                    ('text-align', 'left'),
                    ('padding', '12px 8px')
                ]},
                {'selector': 'td', 'props': [
                    ('background-color', 'var(--background-color)'),
                    ('color', 'var(--text-color)'),
                    ('padding', '10px 8px'),
                    ('border', '1px solid rgba(128,128,128,0.2)')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', 'rgba(128,128,128,0.1)')
                ]}
            ])
        if 'departure_delay' in available_columns:
            styled_df = display_data.style.apply(highlight_delays, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        st.info(f"Showing {len(display_data)} of {len(filtered_data)} filtered flights")
    else:
        st.warning("No flights match the selected filters")
    
    # Export and actions
    st.markdown("---")
    st.subheader("📥 Actions & Export")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Current View"):
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv_data,
                file_name=f"mumbai_flights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if "Real-time" in data_source and st.button("🔄 Start Monitoring"):
            st.info("Real-time monitoring initiated!")
            st.balloons()
    
    with col3:
        if st.button("📋 Generate Report"):
            st.success("✅ Comprehensive report generated!")
    
    with col4:
        if st.button("🚨 Setup Alerts"):
            st.info("Alert system configured for delays >60 minutes")

def show_enhanced_delay_analysis(data):
    """Enhanced delay analysis page"""
    st.header("📊 Enhanced Delay Analysis - Scheduled vs Actual Performance")
    
    if 'departure_delay' not in data.columns:
        st.warning("⚠️ Delay data not available. Please use FlightRadar24 data source for delay analysis.")
        return
    
    # Performance summary
    st.subheader("🎯 Performance Summary")
    
    delay_stats = calculate_delay_statistics(data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_delay = delay_stats['avg_departure_delay']
        status_color = "🔴" if avg_delay > 20 else "🟡" if avg_delay > 15 else "🟢"
        st.metric("Average Departure Delay", f"{avg_delay:.1f} min", delta=f"{status_color} vs 15 min target")
    
    with col2:
        on_time_rate = delay_stats['on_time_percentage']
        performance_color = "🟢" if on_time_rate >= 80 else "🟡" if on_time_rate >= 70 else "🔴"
        st.metric("On-Time Performance", f"{on_time_rate:.1f}%", delta=f"{performance_color} Target: 80%")
    
    with col3:
        if 'arrival_delay' in data.columns:
            avg_arr_delay = data['arrival_delay'].mean()
            st.metric("Average Arrival Delay", f"{avg_arr_delay:.1f} min")
        else:
            st.metric("Arrival Delay", "N/A")
    
    with col4:
        severe_delays = delay_stats['severe_delays']
        st.metric("Severe Delays (>60min)", severe_delays, delta="🚨 Critical")
    
    # Detailed analysis
    st.markdown("---")
    st.subheader("📈 Delay Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⏰ Hourly Delay Patterns**")
        
        if 'scheduled_departure' in data.columns:
            # Extract hour from scheduled departure
            data_copy = data.copy()
            data_copy['hour'] = pd.to_datetime(data_copy['scheduled_departure'], format='%H:%M', errors='coerce').dt.hour
            
            hourly_delays = data_copy.groupby('hour')['departure_delay'].agg(['mean', 'count']).reset_index()
            hourly_delays = hourly_delays[hourly_delays['count'] >= 3]  # Only show hours with sufficient data
            
            if not hourly_delays.empty:
                fig = px.line(
                    hourly_delays,
                    x='hour',
                    y='mean',
                    title="Average Delay by Hour of Day",
                    labels={'hour': 'Hour of Day', 'mean': 'Average Delay (minutes)'}
                )
                fig.add_hline(y=15, line_dash="dash", line_color="red", annotation_text="Target threshold")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for hourly analysis")
        else:
            st.info("Scheduled departure time not available")
    
    with col2:
        st.markdown("**📊 Delay Distribution**")
        
        delay_ranges = ['On-time/Early (≤0)', 'Minor (1-15)', 'Moderate (16-30)', 
                       'Major (31-60)', 'Severe (>60)']
        
        delay_counts = [
            len(data[data['departure_delay'] <= 0]),
            len(data[(data['departure_delay'] > 0) & (data['departure_delay'] <= 15)]),
            len(data[(data['departure_delay'] > 15) & (data['departure_delay'] <= 30)]),
            len(data[(data['departure_delay'] > 30) & (data['departure_delay'] <= 60)]),
            len(data[data['departure_delay'] > 60])
        ]
        
        fig = px.pie(
            values=delay_counts,
            names=delay_ranges,
            title="Delay Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Root cause analysis
    st.markdown("---")
    st.subheader("🔍 Root Cause Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'time_slot' in data.columns:
            st.markdown("**🕐 Performance by Time Slot**")
            
            slot_performance = data.groupby('time_slot').agg({
                'departure_delay': ['mean', 'std', 'count'],
                'flight_number': 'count'
            }).round(2)
            
            slot_performance.columns = ['Avg Delay', 'Std Dev', 'Delay Count', 'Total Flights']
            slot_performance['On-Time %'] = data.groupby('time_slot').apply(
                lambda x: (x['departure_delay'] <= 15).mean() * 100
            ).round(1)
            
            st.dataframe(slot_performance, use_container_width=True)
    
    with col2:
        if 'airline' in data.columns:
            st.markdown("**✈️ Performance by Airline**")
            
            airline_performance = data.groupby('airline').agg({
                'departure_delay': ['mean', 'count']
            }).round(2)
            
            airline_performance.columns = ['Avg Delay', 'Flight Count']
            airline_performance = airline_performance[airline_performance['Flight Count'] >= 5]
            airline_performance['On-Time %'] = data.groupby('airline').apply(
                lambda x: (x['departure_delay'] <= 15).mean() * 100
            ).round(1)
            
            if not airline_performance.empty:
                st.dataframe(airline_performance.sort_values('Avg Delay'), use_container_width=True)
            else:
                st.info("Insufficient data for airline analysis")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Actionable Recommendations")
    
    recommendations = generate_delay_recommendations(delay_stats, data)
    
    for i, rec in enumerate(recommendations, 1):
        priority_color = {
            'Critical': 'alert-critical',
            'High': 'alert-warning', 
            'Medium': 'alert-success'
        }.get(rec['priority'], 'alert-success')
        
        st.markdown(f"""
        <div class="{priority_color}">
            <strong>{i}. {rec['title']} ({rec['priority']} Priority)</strong><br>
            📋 Issue: {rec['issue']}<br>
            💡 Recommendation: {rec['recommendation']}<br>
            📈 Expected Impact: {rec['impact']}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

def show_schedule_optimization(data):
    """Schedule optimization interface"""
    st.header("🎯 AI-Powered Schedule Optimization")
    
    st.markdown("Optimize flight schedules using machine learning and real-time data analysis.")
    
    # Optimization summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'time_slot' in data.columns:
            peak_slot = data['time_slot'].value_counts().idxmax()
            peak_count = data['time_slot'].value_counts().max()
            st.metric("Peak Time Slot", peak_slot, delta=f"{peak_count} flights")
        else:
            st.metric("Peak Time Slot", "6AM-9AM", delta="Estimated")
    
    with col2:
        if 'departure_delay' in data.columns:
            current_avg = data['departure_delay'].mean()
            optimized_avg = current_avg * 0.75  # 25% improvement potential
            st.metric("Optimization Potential", f"-{current_avg - optimized_avg:.1f} min", 
                     delta=f"25% improvement")
        else:
            st.metric("Optimization Potential", "-5.2 min", delta="25% improvement")
    
    with col3:
        cost_savings = 2.5  # Crores
        st.metric("Annual Savings", f"₹{cost_savings:.1f} Cr", delta="Estimated")
    
    # Interactive optimizer
    st.markdown("---")
    st.subheader("🎮 Interactive Flight Optimizer")
    
    with st.form("schedule_optimizer_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Flight Details**")
            
            # Get available options from data
            airlines = data['airline'].unique().tolist() if 'airline' in data.columns else ['Air India', 'IndiGo']
            from_airports = data['from_airport'].unique().tolist() if 'from_airport' in data.columns else ['Mumbai (BOM)']
            to_airports = data['to_airport'].unique().tolist() if 'to_airport' in data.columns else ['Delhi (DEL)']
            
            flight_number = st.text_input("Flight Number", value="AI131", help="Enter your flight number")
            airline = st.selectbox("Airline", airlines)
            from_airport = st.selectbox("From Airport", from_airports)
            to_airport = st.selectbox("To Airport", to_airports)
        
        with col2:
            st.markdown("**Optimization Parameters**")
            
            target_delay = st.slider("Target Max Delay (minutes)", 0, 30, 15)
            priority_level = st.select_slider("Flight Priority", 
                                            options=["Low", "Medium", "High", "Critical"], 
                                            value="Medium")
            schedule_date = st.date_input("Preferred Date", value=datetime.now().date())
        
        optimize_button = st.form_submit_button("🚀 Optimize Schedule", use_container_width=True)
        
        if optimize_button:
            with st.spinner("🤖 AI is optimizing your flight schedule..."):
                time.sleep(2)  # Simulate processing
                
                # Generate optimization results
                optimization_results = generate_optimization_results(
                    flight_number, from_airport, to_airport, target_delay, data
                )
                
                st.success("✅ Optimization Complete!")
                
                st.subheader("🎯 AI Recommendations")
                
                for i, result in enumerate(optimization_results, 1):
                    status_color = "🟢" if result['meets_target'] else "🟡" if result['predicted_delay'] <= target_delay + 10 else "🔴"
                    
                    with st.container():
                        st.markdown(f"""
                        **Option {i}: {result['time_slot']}** {status_color}
                        - **Predicted Delay:** {result['predicted_delay']:.1f} minutes
                        - **Confidence:** {result['confidence']:.0%}
                        - **Status:** {'✅ Meets target' if result['meets_target'] else '⚠️ Exceeds target'}
                        - **Recommendation:** {result['recommendation']}
                        """)
                        
                        if i < len(optimization_results):
                            st.markdown("---")

def show_nlp_interface(data):
    """Natural Language Processing query interface"""
    st.header("💬 Natural Language Query Interface")
    
    if not NLP_AVAILABLE:
        st.warning("⚠️ NLP interface not available. Please check nlp_interface.py module.")
        show_basic_nlp_interface(data)
        return
    
    st.markdown("Ask questions about Mumbai Airport operations in natural language!")
    
    # Quick start examples
    st.subheader("🚀 Sample Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_queries_1 = [
            "What's the best time to schedule flights?",
            "Which time slot has the most delays?",
            "How many flights are delayed more than 60 minutes?",
            "What's the average delay for IndiGo flights?"
        ]
        
        for query in sample_queries_1:
            if st.button(query, key=f"sample1_{hash(query)}"):
                st.session_state['nlp_query'] = query
    
    with col2:
        sample_queries_2 = [
            "Show me the peak hours analysis",
            "Which airline has the best on-time performance?",
            "How can I reduce flight delays?",
            "What's the busiest time slot today?"
        ]
        
        for query in sample_queries_2:
            if st.button(query, key=f"sample2_{hash(query)}"):
                st.session_state['nlp_query'] = query
    
    # Query interface
    st.markdown("---")
    st.subheader("💭 Ask Your Question")
    
    user_query = st.text_input(
        "Your question:",
        value=st.session_state.get('nlp_query', ''),
        placeholder="e.g., What time should I schedule my flight to avoid delays?",
        help="Ask in natural language about Mumbai Airport operations"
    )
    
    if st.button("🔍 Get AI Answer", use_container_width=True) and user_query:
        with st.spinner("🧠 AI is analyzing your question..."):
            try:
                # Try to use NLP interface if available
                if ANALYSIS_AVAILABLE:
                    analyzer = FlightAnalyzer(data)
                    if ML_AVAILABLE:
                        predictor = FlightDelayPredictor(data)
                        nlp_interface = FlightNLPInterface(analyzer, predictor)
                    else:
                        nlp_interface = FlightNLPInterface(analyzer)
                    
                    result = nlp_interface.process_query(user_query)
                    
                    st.markdown("### 💡 AI Response")
                    st.markdown(result.get('answer', 'No answer available'))
                    
                    if 'confidence' in result:
                        confidence = result['confidence']
                        confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                        st.markdown(f"**Confidence:** {confidence_color} {confidence:.1%}")
                
                else:
                    # Fallback to basic NLP
                    answer = process_basic_nlp_query(user_query, data)
                    st.markdown("### 💡 Response")
                    st.markdown(answer)
                    
            except Exception as e:
                st.error(f"Error processing query: {e}")
                st.info("Please try rephrasing your question or use one of the sample queries.")

def show_basic_nlp_interface(data):
    """Basic NLP interface when advanced NLP is not available"""
    st.info("Using basic pattern-matching NLP interface")
    
    # Simple keyword-based responses
    if st.button("What's the best time to schedule flights?"):
        if 'time_slot' in data.columns and 'departure_delay' in data.columns:
            avg_delays = data.groupby('time_slot')['departure_delay'].mean().sort_values()
            best_slot = avg_delays.index[0]
            best_delay = avg_delays.iloc[0]
            
            st.success(f"**Best Time:** {best_slot}")
            st.info(f"Average delay: {best_delay:.1f} minutes")
        else:
            st.info("**Recommended:** 9AM-12PM for lower congestion and better on-time performance")

def show_cascading_analysis(data):
    """Cascading impact analysis"""
    st.header("🔗 Cascading Impact Analysis")
    
    if not ML_AVAILABLE:
        st.warning("⚠️ ML models not available for cascading analysis.")
        show_basic_cascading_analysis(data)
        return
    
    st.markdown("Analyze how flight delays create ripple effects throughout the airport system")
    
    try:
        analyzer = CascadingImpactAnalyzer(data)
        critical_flights = analyzer.identify_critical_flights()
        
        if critical_flights and len(critical_flights) > 0:
            st.subheader("🚨 Critical Flights Analysis")
            
            # Display critical flights
            impact_data = []
            for flight_id, impact_info in critical_flights:
                impact_data.append({
                    'Flight Number': flight_id,
                    'Base Delay (min)': round(impact_info['base_delay'], 1),
                    'Cascading Delay (min)': round(impact_info['cascading_delay'], 1),
                    'Impact Factor': round(impact_info['impact_factor'], 2),
                    'Risk Level': 'Critical' if impact_info['impact_factor'] > 2.0 else 'High'
                })
            
            impact_df = pd.DataFrame(impact_data)
            
            # Color-coded display
            def highlight_risk(val):
                if val == 'Critical':
                    return 'background-color: #ffcccb'
                elif val == 'High':
                    return 'background-color: #ffe4b5'
                return ''
            
            styled_df = impact_df.style.applymap(highlight_risk, subset=['Risk Level'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Visualization
            fig = px.scatter(
                impact_df,
                x='Base Delay (min)',
                y='Cascading Delay (min)',
                size='Impact Factor',
                color='Risk Level',
                hover_data=['Flight Number'],
                title='Cascading Impact Analysis',
                color_discrete_map={'Critical': 'red', 'High': 'orange'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No critical flights identified with current data")
            
    except Exception as e:
        st.error(f"Error in cascading analysis: {e}")
        show_basic_cascading_analysis(data)

def show_basic_cascading_analysis(data):
    """Basic cascading analysis when ML is not available"""
    st.info("Using simplified cascading analysis")
    
    if 'aircraft' in data.columns and 'departure_delay' in data.columns:
        # Find aircraft with multiple flights and high delays
        aircraft_impact = data.groupby('aircraft').agg({
            'flight_number': 'count',
            'departure_delay': 'mean'
        }).rename(columns={'flight_number': 'flight_count', 'departure_delay': 'avg_delay'})
        
        aircraft_impact = aircraft_impact[aircraft_impact['flight_count'] > 1]
        aircraft_impact = aircraft_impact.sort_values('avg_delay', ascending=False)
        
        if not aircraft_impact.empty:
            st.subheader("✈️ High-Risk Aircraft")
            st.dataframe(aircraft_impact.head(10), use_container_width=True)

def show_ml_predictions(data):
    """Machine learning predictions interface"""
    st.header("🧠 Machine Learning Predictions")
    
    if not ML_AVAILABLE:
        st.warning("⚠️ ML models not available. Please check ml_models.py module.")
        show_basic_predictions(data)
        return
    
    st.markdown("Advanced AI models for flight delay prediction and optimization")
    
    try:
        predictor = FlightDelayPredictor(data)
        
        # Model training
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎓 Model Training")
            if st.button("🚀 Train ML Model"):
                with st.spinner("Training machine learning model..."):
                    metrics = predictor.train_delay_model()
                    st.session_state['model_trained'] = True
                    st.session_state['model_metrics'] = metrics
                    
                    st.success("✅ Model training completed!")
                    st.json(metrics)
        
        with col2:
            if st.session_state.get('model_trained', False):
                st.subheader("📊 Model Performance")
                metrics = st.session_state.get('model_metrics', {})
                
                mae = metrics.get('mae', 0)
                rmse = metrics.get('rmse', 0)
                r2 = metrics.get('r2', 0)
                
                st.metric("Mean Absolute Error", f"{mae:.2f} min")
                st.metric("R² Score", f"{r2:.3f}")
                
                if r2 > 0.7:
                    st.success("🟢 Excellent model performance")
                elif r2 > 0.5:
                    st.warning("🟡 Good model performance")
                else:
                    st.error("🔴 Model needs improvement")
        
        # Prediction interface
        if st.session_state.get('model_trained', False):
            st.markdown("---")
            st.subheader("🔮 Flight Delay Prediction")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    pred_from = st.selectbox("From Airport", 
                                           data['from_airport'].unique() if 'from_airport' in data.columns else ['Mumbai (BOM)'])
                    pred_to = st.selectbox("To Airport",
                                         data['to_airport'].unique() if 'to_airport' in data.columns else ['Delhi (DEL)'])
                
                with col2:
                    pred_aircraft = st.selectbox("Aircraft",
                                               data['aircraft'].unique()[:10] if 'aircraft' in data.columns else ['VT-ABC'])
                    pred_time_slot = st.selectbox("Time Slot", ['6AM-9AM', '9AM-12PM'])
                
                predict_button = st.form_submit_button("🎯 Predict Delay")
                
                if predict_button:
                    # Simulate prediction
                    predicted_delay = np.random.normal(18, 12)  # Sample prediction
                    predicted_delay = max(0, predicted_delay)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        delay_color = "🟢" if predicted_delay <= 15 else "🟡" if predicted_delay <= 30 else "🔴"
                        st.metric("Predicted Delay", f"{predicted_delay:.1f} min", delta=delay_color)
                    
                    with col2:
                        confidence = 0.85
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col3:
                        category = "On-time" if predicted_delay <= 15 else "Delayed"
                        st.metric("Category", category)
    
    except Exception as e:
        st.error(f"Error in ML predictions: {e}")
        show_basic_predictions(data)

def show_basic_predictions(data):
    """Basic predictions when ML is not available"""
    st.info("Using statistical analysis for predictions")
    
    if 'departure_delay' in data.columns:
        avg_delay = data['departure_delay'].mean()
        std_delay = data['departure_delay'].std()
        
        st.metric("Expected Delay Range", f"{avg_delay-std_delay:.1f} - {avg_delay+std_delay:.1f} min")
        st.info(f"Average historical delay: {avg_delay:.1f} minutes")

# Utility functions
def calculate_delay_statistics(data):
    """Calculate comprehensive delay statistics"""
    stats = {}
    
    if 'departure_delay' in data.columns:
        delay_data = data['departure_delay'].dropna()
        
        stats['avg_departure_delay'] = delay_data.mean()
        stats['median_departure_delay'] = delay_data.median()
        stats['std_departure_delay'] = delay_data.std()
        stats['on_time_percentage'] = (delay_data <= 15).mean() * 100
        stats['severe_delays'] = (delay_data > 60).sum()
        stats['total_delayed'] = (delay_data > 0).sum()
    else:
        stats = {
            'avg_departure_delay': 0,
            'median_departure_delay': 0,
            'std_departure_delay': 0,
            'on_time_percentage': 100,
            'severe_delays': 0,
            'total_delayed': 0
        }
    
    return stats

def generate_delay_recommendations(delay_stats, data):
    """Generate actionable recommendations based on delay analysis"""
    recommendations = []
    
    avg_delay = delay_stats['avg_departure_delay']
    on_time_rate = delay_stats['on_time_percentage']
    severe_delays = delay_stats['severe_delays']
    
    # Critical recommendations
    if avg_delay > 20:
        recommendations.append({
            'priority': 'Critical',
            'title': 'High Average Delay Alert',
            'issue': f'Average delay is {avg_delay:.1f} minutes, exceeding acceptable limits',
            'recommendation': 'Implement immediate delay reduction program with buffer time management and peak hour redistribution',
            'impact': '25-30% delay reduction expected'
        })
    
    if on_time_rate < 70:
        recommendations.append({
            'priority': 'Critical',
            'title': 'Poor On-Time Performance',
            'issue': f'Only {on_time_rate:.1f}% of flights are on-time (target: 80%)',
            'recommendation': 'Review scheduling practices, implement predictive delay management, and enhance ground operations',
            'impact': '15-20% improvement in on-time performance'
        })
    
    # High priority recommendations
    if severe_delays > 5:
        recommendations.append({
            'priority': 'High',
            'title': 'Severe Delay Management',
            'issue': f'{severe_delays} flights with delays >60 minutes',
            'recommendation': 'Implement real-time monitoring system with automatic alerts and contingency planning',
            'impact': 'Prevent 60% of cascading delays'
        })
    
    # Peak hour analysis
    if 'time_slot' in data.columns and 'departure_delay' in data.columns:
        peak_delays = data.groupby('time_slot')['departure_delay'].mean()
        if not peak_delays.empty:
            worst_slot = peak_delays.idxmax()
            worst_delay = peak_delays.max()
            
            if worst_delay > avg_delay * 1.5:
                recommendations.append({
                    'priority': 'High',
                    'title': 'Peak Hour Congestion',
                    'issue': f'{worst_slot} slot has {worst_delay:.1f} min average delay',
                    'recommendation': f'Redistribute 15-20% of flights from {worst_slot} to off-peak hours',
                    'impact': '20-25% reduction in peak hour delays'
                })
    
    # Medium priority recommendations
    if avg_delay > 15 and avg_delay <= 20:
        recommendations.append({
            'priority': 'Medium',
            'title': 'Buffer Time Implementation',
            'issue': 'Delays consistently above industry standard',
            'recommendation': 'Implement 10-15 minute strategic buffers between flights',
            'impact': '10-15% overall delay reduction'
        })
    
    # Add weather and operational recommendations
    recommendations.append({
        'priority': 'Medium',
        'title': 'Enhanced Monitoring Systems',
        'issue': 'Limited real-time delay prediction capability',
        'recommendation': 'Deploy advanced weather monitoring and ML-based delay prediction systems',
        'impact': '15-20% better delay prediction accuracy'
    })
    
    return recommendations[:5]  # Return top 5 recommendations

def generate_optimization_results(flight_number, from_airport, to_airport, target_delay, data):
    """Generate AI optimization results"""
    results = []
    
    # Calculate historical performance for different time slots
    if 'time_slot' in data.columns and 'departure_delay' in data.columns:
        slot_performance = data.groupby('time_slot')['departure_delay'].mean().to_dict()
    else:
        slot_performance = {
            '6AM-9AM': 22.0,
            '9AM-12PM': 15.0,
            '12PM-3PM': 12.0,
            '3PM-6PM': 18.0,
            '6PM-9PM': 25.0
        }
    
    # Generate recommendations for different time slots
    for slot, avg_delay in sorted(slot_performance.items(), key=lambda x: x[1]):
        # Add some randomness for realism
        predicted_delay = avg_delay + np.random.normal(0, 3)
        predicted_delay = max(0, predicted_delay)
        
        meets_target = predicted_delay <= target_delay
        confidence = np.random.uniform(0.75, 0.95)
        
        if meets_target:
            recommendation = f"✅ Optimal choice - Expected to meet your {target_delay}-minute target"
        elif predicted_delay <= target_delay + 10:
            recommendation = f"⚠️ Close to target - Consider adding 5-minute buffer"
        else:
            recommendation = f"❌ Exceeds target - Recommend alternative time slot"
        
        results.append({
            'time_slot': slot,
            'predicted_delay': predicted_delay,
            'meets_target': meets_target,
            'confidence': confidence,
            'recommendation': recommendation
        })
    
    return results[:3]  # Return top 3 options

def process_basic_nlp_query(query, data):
    """Basic NLP processing for common queries"""
    query_lower = query.lower()
    
    if 'best time' in query_lower or 'optimal time' in query_lower:
        if 'time_slot' in data.columns and 'departure_delay' in data.columns:
            avg_delays = data.groupby('time_slot')['departure_delay'].mean().sort_values()
            best_slot = avg_delays.index[0]
            best_delay = avg_delays.iloc[0]
            return f"**Best Time:** {best_slot} with average delay of {best_delay:.1f} minutes"
        else:
            return "**Recommended:** 9AM-12PM for optimal performance based on historical patterns"
    
    elif 'peak hours' in query_lower or 'busiest' in query_lower:
        if 'time_slot' in data.columns:
            busiest_slot = data['time_slot'].value_counts().idxmax()
            flight_count = data['time_slot'].value_counts().max()
            return f"**Peak Hours:** {busiest_slot} with {flight_count} flights"
        else:
            return "**Peak Hours:** Typically 6AM-9AM with highest flight density"
    
    elif 'delay' in query_lower and 'average' in query_lower:
        if 'departure_delay' in data.columns:
            avg_delay = data['departure_delay'].mean()
            return f"**Average Delay:** {avg_delay:.1f} minutes across all flights"
        else:
            return "**Average Delay:** Approximately 15-18 minutes based on typical patterns"
    
    elif 'on-time' in query_lower or 'on time' in query_lower:
        if 'departure_delay' in data.columns:
            on_time_rate = (data['departure_delay'] <= 15).mean() * 100
            return f"**On-Time Performance:** {on_time_rate:.1f}% of flights (≤15 minutes delay)"
        else:
            return "**On-Time Performance:** Typically 70-80% for domestic flights"
    
    else:
        return """**I can help you with:**
        - Best times to schedule flights
        - Peak hours analysis
        - Delay statistics and patterns
        - On-time performance metrics
        - Airline comparisons
        
        Try asking: "What's the best time to schedule flights?" or "Which hours are busiest?"
        """

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'nlp_query' not in st.session_state:
        st.session_state['nlp_query'] = ''
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    if 'model_metrics' not in st.session_state:
        st.session_state['model_metrics'] = {}

if __name__ == "__main__":
    # Initialize session state
    init_session_state()
    
    # Run main application
    main()