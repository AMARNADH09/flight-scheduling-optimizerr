from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd
import numpy as np

class FlightDelayPredictor:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def prepare_features(self):
        """Prepare features for ML model"""
        features_df = self.data.copy()
        categorical_cols = ['from_airport', 'to_airport', 'aircraft', 'time_slot']
        for col in categorical_cols:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[f'{col}_encoded'] = le.fit_transform(features_df[col].astype(str))
                self.encoders[col] = le
        features_df['hour'] = pd.to_datetime(features_df['scheduled_departure']).dt.hour
        features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        feature_columns = [
            'from_airport_encoded', 'to_airport_encoded', 'aircraft_encoded',
            'time_slot_encoded', 'hour', 'day_of_week', 'is_weekend'
        ]
        X = features_df[feature_columns].fillna(0)
        y = features_df['departure_delay'].fillna(0)
        return X, y
    
    def train_delay_model(self):
        """Train delay prediction model"""
        X, y = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        return metrics
    
    def predict_delay(self, flight_features):
        """Predict delay for a new flight"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        features_scaled = self.scaler.transform([flight_features])
        prediction = self.model.predict(features_scaled)[0]
        return prediction
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            return None
        feature_names = [
            'from_airport', 'to_airport', 'aircraft',
            'time_slot', 'hour', 'day_of_week', 'is_weekend'
        ]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df

    def schedule_optimizer(self, flight_data, target_delay=5):
        """Suggest optimal scheduling to minimize delays"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        recommendations = []
        time_slots = ['6AM-9AM', '9AM-12PM']
        for slot in time_slots:
            test_features = flight_data.copy()
            test_features['time_slot'] = slot
            # Encode categorical features using trained encoders
            encoded = []
            for col in ['from_airport', 'to_airport', 'aircraft', 'time_slot']:
                if col in self.encoders and col in test_features:
                    try:
                        encoded.append(self.encoders[col].transform([test_features[col]])[0])
                    except ValueError:
                        encoded.append(0)  # Unknown category fallback
                else:
                    encoded.append(0)
            # Time-based features
            hour = pd.to_datetime(test_features['scheduled_departure']).hour if 'scheduled_departure' in test_features else 0
            day_of_week = pd.to_datetime(test_features['date']).dayofweek if 'date' in test_features else 0
            is_weekend = int(day_of_week in [5, 6])
            feature_vector = encoded + [hour, day_of_week, is_weekend]
            features_scaled = self.scaler.transform([feature_vector])
            predicted_delay = self.model.predict(features_scaled)[0]
            recommendations.append({
                'time_slot': slot,
                'predicted_delay': predicted_delay,
                'meets_target': predicted_delay <= target_delay
            })
        return sorted(recommendations, key=lambda x: x['predicted_delay'])

class CascadingImpactAnalyzer:
    def __init__(self, data):
        self.data = data
        
    def build_dependency_graph(self):
        """Build flight dependency network"""
        aircraft_schedules = self.data.groupby('aircraft').apply(
            lambda x: x.sort_values('scheduled_departure')
        ).reset_index(drop=True)  # <-- Fix: reset index to remove ambiguity

        cascading_effects = {}
        for aircraft, flights in aircraft_schedules.groupby('aircraft'):
            flights_sorted = flights.sort_values('scheduled_departure')
            cumulative_delay = 0
            for idx, flight in flights_sorted.iterrows():
                current_delay = flight['departure_delay'] + cumulative_delay
                cascading_effects[flight['flight_number']] = {
                    'base_delay': flight['departure_delay'],
                    'cascading_delay': current_delay,
                    'impact_factor': current_delay / max(flight['departure_delay'], 1)
                }
                cumulative_delay = max(0, current_delay - 30)  # 30 min buffer
        return cascading_effects
    
    def identify_critical_flights(self, top_n=10):
        """Identify flights with highest cascading impact"""
        cascading_data = self.build_dependency_graph()
        critical_flights = sorted(
            cascading_data.items(),
            key=lambda x: x[1]['impact_factor'],
            reverse=True
        )[:top_n]
        return critical_flights