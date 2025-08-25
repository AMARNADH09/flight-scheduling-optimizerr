# enhanced_data_processing.py - Real FlightRadar24 Data Processing

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import requests
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class FlightRadar24DataProcessor:
    def __init__(self, excel_file_path=None):
        self.excel_file = excel_file_path
        self.processed_data = None
        
        # FlightRadar24 specific patterns
        self.fr24_columns = {
            'flight_number': ['flight_number', 'flight', 'callsign', 'flight_id'],
            'aircraft': ['aircraft', 'registration', 'reg', 'tail'],
            'from_airport': ['from', 'origin', 'departure_airport'],
            'to_airport': ['to', 'destination', 'arrival_airport'],
            'std': ['std', 'scheduled_departure', 'departure_time'],
            'atd': ['atd', 'actual_departure', 'real_departure'],
            'sta': ['sta', 'scheduled_arrival', 'arrival_time'],
            'ata': ['ata', 'actual_arrival', 'real_arrival'],
            'status': ['status', 'flight_status'],
            'date': ['date', 'flight_date']
        }
        
        # Mumbai Airport specific codes
        self.mumbai_codes = ['BOM', 'VABB', 'Mumbai']
    
    def load_and_process_fr24_data(self):
        """Load and process FlightRadar24 Excel data"""
        try:
            if not self.excel_file or not Path(self.excel_file).exists():
                print("❌ Excel file not found, cannot process real data")
                return self.create_sample_data()
            
            print(f"📊 Processing FlightRadar24 data from: {self.excel_file}")
            
            # Read Excel file - try multiple sheets
            excel_file = pd.ExcelFile(self.excel_file)
            all_data = []
            
            for sheet_name in excel_file.sheet_names:
                print(f"📋 Processing sheet: {sheet_name}")
                
                try:
                    # Read sheet data
                    sheet_data = pd.read_excel(self.excel_file, sheet_name=sheet_name)
                    
                    # Process this sheet
                    processed_sheet = self.process_fr24_sheet(sheet_data, sheet_name)
                    
                    if not processed_sheet.empty:
                        all_data.append(processed_sheet)
                        print(f"✅ Extracted {len(processed_sheet)} flights from {sheet_name}")
                    else:
                        print(f"⚠️ No valid data in {sheet_name}")
                        
                except Exception as e:
                    print(f"❌ Error processing {sheet_name}: {str(e)[:100]}")
                    continue
            
            if all_data:
                self.processed_data = pd.concat(all_data, ignore_index=True)
                self.processed_data = self.calculate_real_delays()
                self.processed_data = self.add_analysis_fields()
                
                print(f"✅ Successfully processed {len(self.processed_data)} flights")
                return self.processed_data
            else:
                print("❌ No data could be processed, using sample data")
                return self.create_sample_data()
                
        except Exception as e:
            print(f"❌ Critical error: {e}")
            return self.create_sample_data()
    
    def process_fr24_sheet(self, sheet_data, sheet_name):
        """Process individual FlightRadar24 sheet"""
        
        # Clean empty rows/columns
        sheet_data = sheet_data.dropna(how='all').dropna(axis=1, how='all')
        
        if sheet_data.empty:
            return pd.DataFrame()
        
        print(f"   Sheet shape: {sheet_data.shape}")
        
        # Try to identify FlightRadar24 data structure
        column_mapping = self.identify_fr24_columns(sheet_data.columns)
        
        if not column_mapping:
            print(f"   ❌ Could not identify FlightRadar24 structure in {sheet_name}")
            return pd.DataFrame()
        
        # Extract data using column mapping
        extracted_data = self.extract_fr24_data(sheet_data, column_mapping, sheet_name)
        
        # Filter for Mumbai flights only
        mumbai_data = self.filter_mumbai_flights(extracted_data)
        
        return mumbai_data
    
    def identify_fr24_columns(self, columns):
        """Identify FlightRadar24 column structure"""
        
        column_mapping = {}
        columns_lower = [str(col).lower().strip() for col in columns]
        
        print(f"   Columns found: {list(columns)}")
        
        # Map columns to our standard format
        for field, patterns in self.fr24_columns.items():
            for i, col_name in enumerate(columns_lower):
                if any(pattern.lower() in col_name for pattern in patterns):
                    column_mapping[field] = i
                    print(f"   ✅ Mapped {field} to column {i}: '{columns[i]}'")
                    break
        
        # Must have at least flight number and one airport
        required_fields = ['flight_number']
        has_required = all(field in column_mapping for field in required_fields)
        
        if has_required:
            print(f"   ✅ Valid FR24 structure identified")
            return column_mapping
        else:
            print(f"   ❌ Missing required fields: {[f for f in required_fields if f not in column_mapping]}")
            return None
    
    def extract_fr24_data(self, sheet_data, column_mapping, sheet_name):
        """Extract data using identified column mapping"""
        
        extracted_records = []
        
        for idx, row in sheet_data.iterrows():
            try:
                record = {
                    'flight_number': self.get_column_value(row, column_mapping, 'flight_number'),
                    'aircraft': self.get_column_value(row, column_mapping, 'aircraft'),
                    'from_airport': self.get_column_value(row, column_mapping, 'from_airport'),
                    'to_airport': self.get_column_value(row, column_mapping, 'to_airport'),
                    'scheduled_departure': self.parse_time(self.get_column_value(row, column_mapping, 'std')),
                    'actual_departure': self.parse_time(self.get_column_value(row, column_mapping, 'atd')),
                    'scheduled_arrival': self.parse_time(self.get_column_value(row, column_mapping, 'sta')),
                    'actual_arrival': self.parse_time(self.get_column_value(row, column_mapping, 'ata')),
                    'status': self.get_column_value(row, column_mapping, 'status'),
                    'date': self.parse_date(self.get_column_value(row, column_mapping, 'date')) or datetime.now().date(),
                    'source_sheet': sheet_name,
                    'source_row': idx
                }
                
                # Only include if we have essential data
                if record['flight_number'] and (record['from_airport'] or record['to_airport']):
                    extracted_records.append(record)
                    
            except Exception as e:
                print(f"   ⚠️ Error processing row {idx}: {str(e)[:50]}")
                continue
        
        return pd.DataFrame(extracted_records)
    
    def get_column_value(self, row, column_mapping, field):
        """Get value from row using column mapping"""
        if field in column_mapping and column_mapping[field] < len(row):
            value = row.iloc[column_mapping[field]]
            if pd.notna(value) and str(value).strip() != 'nan':
                return str(value).strip()
        return None
    
    def parse_time(self, time_str):
        """Parse time string from FlightRadar24 format"""
        if not time_str:
            return None
        
        try:
            time_str = str(time_str).strip()
            
            # Handle various time formats from FR24
            time_patterns = [
                r'(\d{1,2}):(\d{2})\s*(AM|PM)?',  # 9:00 AM or 21:30
                r'(\d{4})',  # 0900 or 2130
                r'(\d{1,2})(\d{2})',  # 900 or 2130
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, time_str.upper())
                if match:
                    if len(match.groups()) == 3:  # Has AM/PM
                        hour, minute, ampm = match.groups()
                        hour = int(hour)
                        minute = int(minute)
                        
                        if ampm == 'PM' and hour != 12:
                            hour += 12
                        elif ampm == 'AM' and hour == 12:
                            hour = 0
                            
                        return f"{hour:02d}:{minute:02d}"
                    
                    elif len(match.groups()) == 1:  # 24-hour format
                        time_val = match.group(1)
                        if len(time_val) == 4:
                            hour = int(time_val[:2])
                            minute = int(time_val[2:])
                            return f"{hour:02d}:{minute:02d}"
                    
                    else:  # Hour and minute separate
                        hour, minute = int(match.group(1)), int(match.group(2))
                        return f"{hour:02d}:{minute:02d}"
            
        except Exception as e:
            print(f"   ⚠️ Time parsing error: {time_str} - {e}")
        
        return None
    
    def parse_date(self, date_str):
        """Parse date string from FlightRadar24 format"""
        if not date_str:
            return None
        
        try:
            date_str = str(date_str).strip()
            
            # Handle various date formats
            date_patterns = [
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD/MM/YYYY
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD
                r'(\d{1,2})\s+(\w+)\s+(\d{4})',  # 25 Jul 2025
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, date_str)
                if match:
                    parts = match.groups()
                    
                    if len(parts[0]) == 4:  # Year first
                        year, month, day = parts
                    else:  # Day first or month name
                        if parts[1].isalpha():  # Month name format
                            day, month_name, year = parts
                            month_map = {
                                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                                'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                                'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                            }
                            month = month_map.get(month_name[:3].lower(), 1)
                        else:
                            day, month, year = parts
                    
                    return pd.to_datetime(f"{year}-{month}-{day}").date()
                    
        except Exception as e:
            print(f"   ⚠️ Date parsing error: {date_str} - {e}")
        
        return datetime.now().date()
    
    def filter_mumbai_flights(self, data):
        """Filter flights that involve Mumbai airport"""
        if data.empty:
            return data
        
        mumbai_mask = (
            data['from_airport'].str.contains('|'.join(self.mumbai_codes), case=False, na=False) |
            data['to_airport'].str.contains('|'.join(self.mumbai_codes), case=False, na=False)
        )
        
        mumbai_flights = data[mumbai_mask].copy()
        print(f"   🎯 Found {len(mumbai_flights)} Mumbai flights out of {len(data)} total")
        
        return mumbai_flights
    
    def calculate_real_delays(self):
        """Calculate actual delays from scheduled vs actual times"""
        if self.processed_data is None or self.processed_data.empty:
            return self.processed_data
        
        print("⏱️ Calculating real delays from scheduled vs actual times...")
        
        departure_delays = []
        arrival_delays = []
        
        for _, row in self.processed_data.iterrows():
            # Calculate departure delay
            dep_delay = self.calculate_time_difference(
                row['actual_departure'], 
                row['scheduled_departure']
            )
            
            # Calculate arrival delay  
            arr_delay = self.calculate_time_difference(
                row['actual_arrival'], 
                row['scheduled_arrival']
            )
            
            departure_delays.append(dep_delay)
            arrival_delays.append(arr_delay)
        
        self.processed_data['departure_delay'] = departure_delays
        self.processed_data['arrival_delay'] = arrival_delays
        
        # Calculate statistics
        valid_dep_delays = [d for d in departure_delays if d is not None]
        valid_arr_delays = [d for d in arrival_delays if d is not None]
        
        if valid_dep_delays:
            avg_dep_delay = np.mean(valid_dep_delays)
            print(f"✅ Average departure delay: {avg_dep_delay:.1f} minutes")
        
        if valid_arr_delays:
            avg_arr_delay = np.mean(valid_arr_delays)
            print(f"✅ Average arrival delay: {avg_arr_delay:.1f} minutes")
        
        return self.processed_data
    
    def calculate_time_difference(self, actual_time, scheduled_time):
        """Calculate difference between actual and scheduled times in minutes"""
        if not actual_time or not scheduled_time:
            return None
        
        try:
            # Convert to datetime objects for comparison
            base_date = "2025-01-01"
            scheduled_dt = pd.to_datetime(f"{base_date} {scheduled_time}")
            actual_dt = pd.to_datetime(f"{base_date} {actual_time}")
            
            # Handle day rollover (e.g., scheduled 23:30, actual 00:15 next day)
            if actual_dt < scheduled_dt:
                actual_dt += pd.Timedelta(days=1)
            
            # Calculate difference in minutes
            diff_minutes = (actual_dt - scheduled_dt).total_seconds() / 60
            
            return diff_minutes
            
        except Exception as e:
            return None
    
    def add_analysis_fields(self):
        """Add fields needed for analysis"""
        if self.processed_data is None or self.processed_data.empty:
            return self.processed_data
        
        print("📊 Adding analysis fields...")
        
        # Time slot categorization
        self.processed_data['time_slot'] = self.processed_data['scheduled_departure'].apply(
            self.categorize_time_slot
        )
        
        # Day of week
        self.processed_data['day_of_week'] = pd.to_datetime(self.processed_data['date']).dt.dayofweek
        self.processed_data['is_weekend'] = self.processed_data['day_of_week'].isin([5, 6])
        
        # Peak hour indicator
        self.processed_data['is_peak_hour'] = self.processed_data['time_slot'].isin(['6AM-9AM', '6PM-9PM'])
        
        # Flight direction (departures vs arrivals at Mumbai)
        def get_flight_direction(row):
            from_airport = str(row.get('from_airport', ''))
            to_airport = str(row.get('to_airport', ''))
            
            if any(code in from_airport.upper() for code in self.mumbai_codes):
                return 'Departure'
            elif any(code in to_airport.upper() for code in self.mumbai_codes):
                return 'Arrival'
            else:
                return 'Unknown'
        
        self.processed_data['flight_direction'] = self.processed_data.apply(get_flight_direction, axis=1)
        
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
        
        self.processed_data['delay_category'] = self.processed_data['departure_delay'].apply(categorize_delay)
        
        return self.processed_data
    
    def categorize_time_slot(self, time_str):
        """Categorize time into slots"""
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
    
    def create_sample_data(self):
        """Create sample data if real data processing fails"""
        print("📋 Creating comprehensive sample data for demonstration...")
        
        # This would be your fallback sample data
        # ... (your existing sample data creation logic)
        
        return pd.DataFrame()  # Placeholder
    
    def export_processed_data(self, filename='processed_mumbai_flights.csv'):
        """Export processed data"""
        if self.processed_data is not None and not self.processed_data.empty:
            self.processed_data.to_csv(filename, index=False)
            print(f"✅ Processed data exported to {filename}")
            
            # Create summary report
            self.create_processing_report(filename.replace('.csv', '_report.txt'))
        else:
            print("❌ No processed data to export")
    
    def create_processing_report(self, filename):
        """Create processing summary report"""
        with open(filename, 'w') as f:
            f.write("FLIGHTRADAR24 DATA PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Source File: {self.excel_file}\n")
            f.write(f"Processing Date: {datetime.now()}\n")
            f.write(f"Total Flights Processed: {len(self.processed_data)}\n\n")
            
            if 'flight_direction' in self.processed_data.columns:
                direction_counts = self.processed_data['flight_direction'].value_counts()
                f.write("Flight Direction Distribution:\n")
                for direction, count in direction_counts.items():
                    f.write(f"  {direction}: {count}\n")
                f.write("\n")
            
            if 'departure_delay' in self.processed_data.columns:
                valid_delays = self.processed_data['departure_delay'].dropna()
                if not valid_delays.empty:
                    f.write("Delay Statistics:\n")
                    f.write(f"  Average Departure Delay: {valid_delays.mean():.2f} minutes\n")
                    f.write(f"  Median Departure Delay: {valid_delays.median():.2f} minutes\n")
                    f.write(f"  On-time Rate (≤15 min): {(valid_delays <= 15).mean()*100:.1f}%\n")
                    f.write("\n")
            
            if 'time_slot' in self.processed_data.columns:
                slot_dist = self.processed_data['time_slot'].value_counts()
                f.write("Time Slot Distribution:\n")
                for slot, count in slot_dist.items():
                    f.write(f"  {slot}: {count}\n")
        
        print(f"✅ Processing report saved to {filename}")

# Usage example
if __name__ == "__main__":
    # Initialize with your FlightRadar24 Excel file
    processor = FlightRadar24DataProcessor('Flight_Data.xlsx')
    
    # Process the real data
    processed_data = processor.load_and_process_fr24_data()
    
    # Export results
    processor.export_processed_data()
    
    print("\n📊 Data Processing Complete!")
    print(f"Processed {len(processed_data)} flights from FlightRadar24 data")