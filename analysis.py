# src/analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

class FlightAnalyzer:
    def __init__(self, processed_data):
        self.data = processed_data
        
    def analyze_peak_hours(self):
        """Find busiest time slots"""
        if 'time_slot' not in self.data.columns:
            return {
                'busiest_slot': '6AM-9AM',
                'peak_flights': 0,
                'hourly_distribution': {}
            }
            
        hourly_traffic = self.data['time_slot'].value_counts()
        
        peak_analysis = {
            'busiest_slot': hourly_traffic.idxmax() if len(hourly_traffic) > 0 else 'Unknown',
            'peak_flights': hourly_traffic.max() if len(hourly_traffic) > 0 else 0,
            'hourly_distribution': hourly_traffic.to_dict()
        }
        return peak_analysis
    
    def analyze_delays(self):
        """Comprehensive delay analysis"""
        delay_cols = ['departure_delay', 'arrival_delay']
        existing_delay_cols = [col for col in delay_cols if col in self.data.columns]
        
        if not existing_delay_cols:
            return {
                'avg_departure_delay': 0,
                'avg_arrival_delay': 0,
                'delay_correlation': 0,
                'worst_delay_routes': pd.Series(dtype=float)
            }
        
        delay_stats = {}
        
        if 'departure_delay' in self.data.columns:
            delay_stats['avg_departure_delay'] = self.data['departure_delay'].mean()
        else:
            delay_stats['avg_departure_delay'] = 0
            
        if 'arrival_delay' in self.data.columns:
            delay_stats['avg_arrival_delay'] = self.data['arrival_delay'].mean()
        else:
            delay_stats['avg_arrival_delay'] = 0
        
        # Delay correlation
        if len(existing_delay_cols) == 2:
            delay_stats['delay_correlation'] = self.data['departure_delay'].corr(self.data['arrival_delay'])
        else:
            delay_stats['delay_correlation'] = 0
        
        # Worst delay routes
        if 'departure_delay' in self.data.columns and 'from_airport' in self.data.columns and 'to_airport' in self.data.columns:
            route_delays = self.data.groupby(['from_airport', 'to_airport'])['departure_delay'].mean()
            delay_stats['worst_delay_routes'] = route_delays.sort_values(ascending=False).head(10)
        else:
            delay_stats['worst_delay_routes'] = pd.Series(dtype=float)
        
        # Additional statistics
        if 'departure_delay' in self.data.columns:
            delay_stats['delay_std'] = self.data['departure_delay'].std()
            delay_stats['delay_median'] = self.data['departure_delay'].median()
            delay_stats['on_time_percentage'] = (self.data['departure_delay'] <= 15).mean() * 100
        
        return delay_stats
    
    def identify_cascading_flights(self):
        """Identify flights with highest cascading impact"""
        if 'departure_delay' not in self.data.columns or 'flight_number' not in self.data.columns:
            return pd.Series(dtype=float)
        
        # Calculate impact score based on delay frequency and magnitude
        flight_stats = self.data.groupby('flight_number').agg({
            'departure_delay': ['mean', 'std', 'count']
        }).fillna(0)
        
        # Flatten column names
        flight_stats.columns = ['delay_mean', 'delay_std', 'flight_count']
        
        # Impact score: high delay * high frequency * high variability
        impact_score = (
            flight_stats['delay_mean'] * 
            flight_stats['flight_count'] * 
            (flight_stats['delay_std'] + 1)  # Add 1 to avoid zero multiplication
        )
        
        high_impact_flights = impact_score.sort_values(ascending=False).head(20)
        return high_impact_flights
    
    def optimal_scheduling_windows(self):
        """Find optimal time windows for minimal delays"""
        if 'time_slot' not in self.data.columns:
            return pd.DataFrame()
        
        delay_cols = ['departure_delay', 'arrival_delay']
        existing_delay_cols = [col for col in delay_cols if col in self.data.columns]
        
        if not existing_delay_cols:
            return pd.DataFrame()
        
        # Group by time slot and calculate statistics
        agg_dict = {}
        for col in existing_delay_cols:
            agg_dict[col] = ['mean', 'std', 'count', 'median']
        
        time_delay_analysis = self.data.groupby('time_slot').agg(agg_dict).round(2)
        
        return time_delay_analysis
    
    def analyze_weather_impact(self):
        """Analyze weather-related delay patterns (simulated for demo)"""
        # Since we don't have weather data, simulate the analysis
        if 'departure_delay' not in self.data.columns:
            return {}
        
        # Simulate weather impact by day of week (weekend = better weather)
        weather_impact = {}
        
        if 'day_of_week' in self.data.columns:
            daily_delays = self.data.groupby('day_of_week')['departure_delay'].mean()
            weather_impact['daily_pattern'] = daily_delays.to_dict()
            weather_impact['worst_weather_day'] = daily_delays.idxmax()
            weather_impact['best_weather_day'] = daily_delays.idxmin()
        
        return weather_impact
    
    def runway_capacity_analysis(self):
        """Analyze runway capacity constraints"""
        if 'time_slot' not in self.data.columns:
            return {}
        
        # Calculate flights per hour for capacity analysis
        slot_counts = self.data['time_slot'].value_counts()
        
        # Assume runway capacity (flights per hour)
        RUNWAY_CAPACITY = 25  # flights per hour per runway
        TOTAL_RUNWAYS = 2    # Mumbai has 2 runways
        
        capacity_analysis = {
            'total_runway_capacity': RUNWAY_CAPACITY * TOTAL_RUNWAYS,
            'current_utilization': slot_counts.to_dict(),
            'capacity_utilization_pct': {},
            'congestion_risk': {}
        }
        
        # Calculate utilization percentages
        for slot, count in slot_counts.items():
            flights_per_hour = count / 3  # 3-hour slots
            utilization_pct = (flights_per_hour / (RUNWAY_CAPACITY * TOTAL_RUNWAYS)) * 100
            capacity_analysis['capacity_utilization_pct'][slot] = round(utilization_pct, 1)
            capacity_analysis['congestion_risk'][slot] = 'High' if utilization_pct > 80 else 'Medium' if utilization_pct > 60 else 'Low'
        
        return capacity_analysis
    
    def generate_scheduling_recommendations(self):
        """Generate actionable scheduling recommendations"""
        recommendations = []
        
        # Peak hours analysis
        peak_data = self.analyze_peak_hours()
        if peak_data['peak_flights'] > 0:
            recommendations.append({
                'category': 'Peak Hours',
                'recommendation': f"Avoid scheduling during {peak_data['busiest_slot']} ({peak_data['peak_flights']} flights). Consider redistributing to off-peak hours.",
                'priority': 'High',
                'impact': 'Reduce congestion by 20-30%'
            })
        
        # Delay analysis
        if 'departure_delay' in self.data.columns:
            delay_stats = self.analyze_delays()
            if delay_stats['avg_departure_delay'] > 15:
                recommendations.append({
                    'category': 'Delay Reduction',
                    'recommendation': f"Average delay is {delay_stats['avg_departure_delay']:.1f} minutes. Implement 15-minute buffer zones between flights.",
                    'priority': 'High',
                    'impact': 'Reduce average delays by 25%'
                })
        
        # Capacity recommendations
        capacity_data = self.runway_capacity_analysis()
        for slot, risk in capacity_data['congestion_risk'].items():
            if risk == 'High':
                recommendations.append({
                    'category': 'Capacity Management',
                    'recommendation': f"Runway utilization in {slot} is at risk. Consider alternate airports or time slots.",
                    'priority': 'Medium',
                    'impact': 'Prevent cascading delays'
                })
        
        return recommendations
    
    def create_visualizations(self):
        """Generate comprehensive visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Delay distribution
            if 'departure_delay' in self.data.columns:
                axes[0,0].hist(self.data['departure_delay'], bins=30, alpha=0.7, color='skyblue')
                axes[0,0].set_title('Departure Delay Distribution')
                axes[0,0].set_xlabel('Delay (minutes)')
                axes[0,0].set_ylabel('Frequency')
            else:
                axes[0,0].text(0.5, 0.5, 'No delay data available', ha='center', va='center')
                axes[0,0].set_title('Departure Delay Distribution')
            
            # 2. Delays by time slot
            if 'time_slot' in self.data.columns and 'departure_delay' in self.data.columns:
                slot_data = [self.data[self.data['time_slot'] == slot]['departure_delay'].dropna() 
                           for slot in self.data['time_slot'].unique()]
                axes[0,1].boxplot(slot_data, labels=self.data['time_slot'].unique())
                axes[0,1].set_title('Delays by Time Slot')
                axes[0,1].set_ylabel('Delay (minutes)')
            else:
                axes[0,1].text(0.5, 0.5, 'No time slot data available', ha='center', va='center')
                axes[0,1].set_title('Delays by Time Slot')
            
            # 3. Flight frequency by hour
            if 'time_slot' in self.data.columns:
                time_counts = self.data['time_slot'].value_counts()
                axes[1,0].bar(time_counts.index, time_counts.values, color='lightgreen')
                axes[1,0].set_title('Flight Frequency by Time Slot')
                axes[1,0].set_ylabel('Number of Flights')
                axes[1,0].tick_params(axis='x', rotation=45)
            else:
                axes[1,0].text(0.5, 0.5, 'No time slot data available', ha='center', va='center')
                axes[1,0].set_title('Flight Frequency by Time Slot')
            
            # 4. Correlation heatmap
            delay_columns = [col for col in ['departure_delay', 'arrival_delay'] if col in self.data.columns]
            if len(delay_columns) >= 2:
                correlation_data = self.data[delay_columns].corr()
                im = axes[1,1].imshow(correlation_data, cmap='coolwarm', aspect='auto')
                axes[1,1].set_xticks(range(len(correlation_data.columns)))
                axes[1,1].set_yticks(range(len(correlation_data.columns)))
                axes[1,1].set_xticklabels(correlation_data.columns)
                axes[1,1].set_yticklabels(correlation_data.columns)
                axes[1,1].set_title('Delay Correlation Matrix')
                
                # Add correlation values
                for i in range(len(correlation_data)):
                    for j in range(len(correlation_data.columns)):
                        text = axes[1,1].text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                                            ha="center", va="center", color="black")
            else:
                axes[1,1].text(0.5, 0.5, 'Insufficient delay data for correlation', ha='center', va='center')
                axes[1,1].set_title('Delay Correlation Matrix')
            
            plt.tight_layout()
            plt.savefig('flight_analysis.png', dpi=300, bbox_inches='tight')
            return fig
        
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return None
    
    def export_analysis_report(self, filename='flight_analysis_report.txt'):
        """Export comprehensive analysis report"""
        try:
            with open(filename, 'w') as f:
                f.write("MUMBAI AIRPORT FLIGHT SCHEDULING ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Peak hours analysis
                peak_data = self.analyze_peak_hours()
                f.write("PEAK HOURS ANALYSIS:\n")
                f.write(f"Busiest Time Slot: {peak_data['busiest_slot']}\n")
                f.write(f"Peak Traffic: {peak_data['peak_flights']} flights\n\n")
                
                # Delay analysis
                delay_stats = self.analyze_delays()
                f.write("DELAY ANALYSIS:\n")
                f.write(f"Average Departure Delay: {delay_stats['avg_departure_delay']:.2f} minutes\n")
                f.write(f"Average Arrival Delay: {delay_stats['avg_arrival_delay']:.2f} minutes\n")
                f.write(f"Delay Correlation: {delay_stats['delay_correlation']:.3f}\n\n")
                
                # Capacity analysis
                capacity_data = self.runway_capacity_analysis()
                f.write("RUNWAY CAPACITY ANALYSIS:\n")
                f.write(f"Total Runway Capacity: {capacity_data['total_runway_capacity']} flights/hour\n")
                for slot, utilization in capacity_data['capacity_utilization_pct'].items():
                    risk = capacity_data['congestion_risk'][slot]
                    f.write(f"{slot}: {utilization}% utilization - {risk} risk\n")
                f.write("\n")
                
                # Recommendations
                recommendations = self.generate_scheduling_recommendations()
                f.write("SCHEDULING RECOMMENDATIONS:\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec['category']} ({rec['priority']} Priority)\n")
                    f.write(f"   Recommendation: {rec['recommendation']}\n")
                    f.write(f"   Expected Impact: {rec['impact']}\n\n")
            
            print(f"Analysis report exported to {filename}")
        
        except Exception as e:
            print(f"Error exporting report: {e}")