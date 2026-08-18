# =============================================================================
# FILE: Scr/data_preprocessing/feature_engineering.py (VERSION 4.0 - PROFESSIONAL)
# =============================================================================
# PURPOSE: Advanced feature engineering for RL state construction with
#          support for multiple state types (classical, LSTM, GRU).
# =============================================================================

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging
import warnings
import sys
import os

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import (
    NETWORK_TYPES, AREA_TYPES, CORE_FEATURES, NTN_ONLY_FEATURES,
    UE_FEATURES, ENV_FEATURES, NUM_NETWORKS, NUM_AREAS,
    FAIR_COMPARISON_FEATURES, NTN_COMPARISON_FEATURES,
    AREA_COLUMN, NETWORK_COLUMN, AVAILABLE_COLUMN
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced Feature Engineer for Hybrid Network Recommendation.
    
    Features:
    - Multiple scaling methods (Standard, Robust, MinMax)
    - Categorical encoding (One-Hot, Label)
    - Missing value imputation
    - Interaction feature creation
    - State construction for RL agents
    - Fair comparison feature extraction
    """
    
    def __init__(
        self, 
        use_scaling: bool = True,
        scaler_type: str = 'robust',
        impute_missing: bool = True,
        create_interactions: bool = True,
        add_polynomial: bool = False,
        n_components: int = 20
    ):
        """
        Initialize the feature engineer.
        
        Args:
            use_scaling: Whether to scale numeric features
            scaler_type: Type of scaler ('standard', 'robust', 'minmax')
            impute_missing: Whether to impute missing values
            create_interactions: Whether to create interaction features
            add_polynomial: Whether to add polynomial features
            n_components: Number of features to keep
        """
        self.use_scaling = use_scaling
        self.scaler_type = scaler_type
        self.impute_missing = impute_missing
        self.create_interactions = create_interactions
        self.add_polynomial = add_polynomial
        self.n_components = n_components
        
        # Initialize scaler
        if use_scaling:
            self.scaler = self._create_scaler(scaler_type)
        else:
            self.scaler = None
        
        # Initialize encoders
        self.area_encoder = LabelEncoder()
        self.network_encoder = LabelEncoder()
        
        # Initialize imputer
        self.imputer = SimpleImputer(strategy='median') if impute_missing else None
        
        # Initialize polynomial features
        if add_polynomial:
            from sklearn.preprocessing import PolynomialFeatures
            self.polynomial = PolynomialFeatures(degree=2, include_bias=False)
        else:
            self.polynomial = None
        
        self.is_fitted = False
        self.feature_names = []
        self.numeric_cols = []
        self.categorical_cols = []
        
        # Feature dimensions
        self.feature_dims = {
            'area': NUM_AREAS,                      # 6
            'available_networks': NUM_NETWORKS,     # 5
            'measurements': NUM_NETWORKS * 3,       # 15
            'ue_features': len(UE_FEATURES),        # 1
            'core_features': len(CORE_FEATURES),    # 9
            'ntn_features': len(NTN_ONLY_FEATURES), # 5
            'total_classical': 27                   # 6 + 5 + 15 + 1
        }
        
        logger.info(f"FeatureEngineer initialized")
        logger.info(f"   Scaling: {use_scaling} ({scaler_type})")
        logger.info(f"   Imputation: {impute_missing}")
        logger.info(f"   Interactions: {create_interactions}")
        logger.info(f"   Polynomial: {add_polynomial}")
        logger.info(f"   Feature dims: {self.feature_dims}")
    
    def _create_scaler(self, scaler_type: str):
        """Create scaler based on type."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the engineer and transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        logger.info("Fitting and transforming features...")
        
        # Create a copy
        df_transformed = df.copy()
        
        # Identify column types
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Encode Area
        if AREA_COLUMN in df.columns:
            try:
                df_transformed['Area_Encoded'] = self.area_encoder.fit_transform(df[AREA_COLUMN])
                logger.info("Area encoded")
            except Exception as e:
                logger.warning(f"Area encoding failed: {e}")
                df_transformed['Area_Encoded'] = 0
        
        # 2. Encode Network Type
        if NETWORK_COLUMN in df.columns:
            try:
                df_transformed['Network_Encoded'] = self.network_encoder.fit_transform(df[NETWORK_COLUMN])
                logger.info("Network encoded")
            except Exception as e:
                logger.warning(f"Network encoding failed: {e}")
                df_transformed['Network_Encoded'] = 0
        
        # 3. Encode Available Networks (One-Hot)
        available_encoded = self._encode_available_networks(df)
        df_transformed = pd.concat([df_transformed, available_encoded], axis=1)
        logger.info(f"Available networks encoded ({len(available_encoded.columns)} columns)")
        
        # 4. Prepare numeric columns for scaling
        numeric_cols_for_scaling = self._get_numeric_cols(df_transformed)
        
        # 5. Impute missing values
        if self.impute_missing and numeric_cols_for_scaling:
            numeric_data = df_transformed[numeric_cols_for_scaling].copy()
            self.imputer.fit(numeric_data)
            imputed_data = self.imputer.transform(numeric_data)
            df_transformed[numeric_cols_for_scaling] = imputed_data
            logger.info(f"Missing values imputed ({len(numeric_cols_for_scaling)} columns)")
        
        # 6. Scale numeric features
        if self.use_scaling and numeric_cols_for_scaling:
            scaled_data = self.scaler.fit_transform(df_transformed[numeric_cols_for_scaling])
            
            # Add scaled columns with suffix
            for i, col in enumerate(numeric_cols_for_scaling):
                df_transformed[f'{col}_scaled'] = scaled_data[:, i]
            
            logger.info(f"Scaled {len(numeric_cols_for_scaling)} numeric features")
        
        # 7. Create interaction features
        if self.create_interactions:
            df_transformed = self._create_interaction_features(df_transformed)
            logger.info("Created interaction features")
        
        # 8. Add polynomial features
        if self.add_polynomial and numeric_cols_for_scaling:
            poly_data = self.polynomial.fit_transform(df_transformed[numeric_cols_for_scaling])
            poly_cols = self.polynomial.get_feature_names_out(numeric_cols_for_scaling)
            
            # Limit to top components
            poly_df = pd.DataFrame(poly_data, columns=poly_cols)
            if poly_df.shape[1] > self.n_components:
                poly_df = poly_df.iloc[:, :self.n_components]
            
            df_transformed = pd.concat([df_transformed, poly_df.add_prefix('poly_')], axis=1)
            logger.info(f"Added polynomial features ({poly_df.shape[1]} columns)")
        
        self.is_fitted = True
        self.feature_names = df_transformed.columns.tolist()
        
        logger.info(f"✅ Transformation complete: {df_transformed.shape}")
        
        return df_transformed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted first. Call fit_transform() first.")
        
        logger.info("Transforming features...")
        
        # Create a copy
        df_transformed = df.copy()
        
        # 1. Encode Area
        if AREA_COLUMN in df.columns:
            try:
                df_transformed['Area_Encoded'] = self.area_encoder.transform(df[AREA_COLUMN])
            except Exception as e:
                logger.warning(f"Area transform failed: {e}")
                df_transformed['Area_Encoded'] = 0
        
        # 2. Encode Network Type
        if NETWORK_COLUMN in df.columns:
            try:
                df_transformed['Network_Encoded'] = self.network_encoder.transform(df[NETWORK_COLUMN])
            except Exception as e:
                logger.warning(f"Network transform failed: {e}")
                df_transformed['Network_Encoded'] = 0
        
        # 3. Encode Available Networks
        available_encoded = self._encode_available_networks(df)
        df_transformed = pd.concat([df_transformed, available_encoded], axis=1)
        
        # 4. Get numeric columns
        numeric_cols_for_scaling = self._get_numeric_cols(df_transformed)
        
        # 5. Impute missing values
        if self.impute_missing and numeric_cols_for_scaling:
            numeric_data = df_transformed[numeric_cols_for_scaling].copy()
            imputed_data = self.imputer.transform(numeric_data)
            df_transformed[numeric_cols_for_scaling] = imputed_data
        
        # 6. Scale numeric features
        if self.use_scaling and numeric_cols_for_scaling:
            scaled_data = self.scaler.transform(df_transformed[numeric_cols_for_scaling])
            
            for i, col in enumerate(numeric_cols_for_scaling):
                df_transformed[f'{col}_scaled'] = scaled_data[:, i]
        
        # 7. Create interaction features
        if self.create_interactions:
            df_transformed = self._create_interaction_features(df_transformed)
        
        # 8. Add polynomial features
        if self.add_polynomial and numeric_cols_for_scaling:
            poly_data = self.polynomial.transform(df_transformed[numeric_cols_for_scaling])
            poly_cols = self.polynomial.get_feature_names_out(numeric_cols_for_scaling)
            poly_df = pd.DataFrame(poly_data, columns=poly_cols)
            if poly_df.shape[1] > self.n_components:
                poly_df = poly_df.iloc[:, :self.n_components]
            df_transformed = pd.concat([df_transformed, poly_df.add_prefix('poly_')], axis=1)
        
        logger.info(f"✅ Transformation complete: {df_transformed.shape}")
        
        return df_transformed
    
    def _encode_available_networks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode available networks as one-hot vectors.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: One-hot encoded available networks
        """
        encoded = np.zeros((len(df), NUM_NETWORKS))
        
        if AVAILABLE_COLUMN not in df.columns:
            columns = [f'Available_{net}' for net in NETWORK_TYPES]
            return pd.DataFrame(encoded, columns=columns, index=df.index)
        
        for idx, row in df.iterrows():
            available_str = row[AVAILABLE_COLUMN]
            if pd.isna(available_str):
                continue
            
            try:
                # Clean and split
                available = [n.strip() for n in str(available_str).split(',')]
                for j, network in enumerate(NETWORK_TYPES):
                    if network in available:
                        encoded[idx, j] = 1
            except Exception as e:
                logger.debug(f"Error encoding row {idx}: {e}")
                continue
        
        columns = [f'Available_{net}' for net in NETWORK_TYPES]
        return pd.DataFrame(encoded, columns=columns, index=df.index)
    
    def _get_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Get numeric columns for scaling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List[str]: Numeric column names
        """
        # Exclude non-numeric and identifier columns
        exclude_cols = [
            'UE_ID', 'network_type', 'Area', 'Available_Networks',
            'Area_Encoded', 'Network_Encoded'
        ]
        
        # Add encoded columns
        exclude_cols.extend([f'Available_{net}' for net in NETWORK_TYPES])
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Exclude already scaled columns
        numeric_cols = [col for col in numeric_cols if not col.endswith('_scaled')]
        numeric_cols = [col for col in numeric_cols if not col.startswith('poly_')]
        
        return numeric_cols
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df_copy = df.copy()
        
        # SNR × SINR interaction
        if 'SNR_dB_scaled' in df.columns and 'SINR_dB_scaled' in df.columns:
            df_copy['SNR_SINR_Product'] = df['SNR_dB_scaled'] * df['SINR_dB_scaled']
        
        # Throughput / (Latency + 1) ratio
        if 'Throughput_Mbps_scaled' in df.columns and 'Latency_ms_scaled' in df.columns:
            df_copy['Throughput_Latency_Ratio'] = df['Throughput_Mbps_scaled'] / (df['Latency_ms_scaled'] + 0.001)
        
        # Quality composite score
        quality_cols = ['SNR_dB_scaled', 'SINR_dB_scaled', 'RSSI_dBm_scaled']
        if all(col in df.columns for col in quality_cols):
            df_copy['Quality_Composite'] = (
                df['SNR_dB_scaled'] * 0.4 + 
                df['SINR_dB_scaled'] * 0.4 + 
                df['RSSI_dBm_scaled'] * 0.2
            )
        
        # SNR × Throughput
        if 'SNR_dB_scaled' in df.columns and 'Throughput_Mbps_scaled' in df.columns:
            df_copy['SNR_Throughput'] = df['SNR_dB_scaled'] * df['Throughput_Mbps_scaled']
        
        # Link quality × Speed (mobility impact)
        if 'Link_Quality_Index_scaled' in df.columns and 'speed_ms_scaled' in df.columns:
            df_copy['Quality_Speed_Interaction'] = df['Link_Quality_Index_scaled'] * (1 - df['speed_ms_scaled'])
        
        return df_copy
    
    def build_classical_state(self, row: pd.Series) -> np.ndarray:
        """
        Build classical (POMDP) state vector from a single row.
        
        State composition:
        - Area (One-Hot): 6 dimensions
        - Available Networks (One-Hot): 5 dimensions
        - Network Measurements: 15 dimensions (5 networks × 3)
        - UE Features: 1 dimension
        
        Returns:
            np.ndarray: State vector of length 27
        """
        # 1. Area (One-Hot)
        area_one_hot = self._one_hot_area(row.get(AREA_COLUMN, 'Unknown'))
        
        # 2. Available Networks (One-Hot)
        available_one_hot = self._one_hot_available(row.get(AVAILABLE_COLUMN, ''))
        
        # 3. Network measurements
        measurements = self._get_measurements(row)
        
        # 4. UE features
        ue_state = self._get_ue_state(row)
        
        # Combine all features
        state = np.concatenate([
            area_one_hot,       # 6
            available_one_hot,  # 5
            measurements,       # 15
            ue_state           # 1
        ])
        
        return state.astype(np.float32)
    
    def _one_hot_area(self, area: str) -> np.ndarray:
        """Convert area to one-hot encoding."""
        one_hot = np.zeros(NUM_AREAS)
        if pd.isna(area):
            return one_hot
        
        area = str(area).strip()
        if area in AREA_TYPES:
            one_hot[AREA_TYPES.index(area)] = 1
        return one_hot
    
    def _one_hot_available(self, available: str) -> np.ndarray:
        """Convert available networks to one-hot encoding."""
        one_hot = np.zeros(NUM_NETWORKS)
        if pd.isna(available):
            return one_hot
        
        try:
            networks = [n.strip() for n in str(available).split(',')]
            for i, network in enumerate(NETWORK_TYPES):
                if network in networks:
                    one_hot[i] = 1
        except Exception:
            pass
        
        return one_hot
    
    def _get_measurements(self, row: pd.Series) -> np.ndarray:
        """
        Extract network measurements from a row.
        
        Measurements: SNR_dB, SINR_dB, RSSI_dBm
        """
        measurements = [
            row.get('SNR_dB', 0.0),
            row.get('SINR_dB', 0.0),
            row.get('RSSI_dBm', 0.0)
        ]
        
        # Convert to float and handle NaN
        measurements = [float(m) if not pd.isna(m) else 0.0 for m in measurements]
        
        # Repeat for all networks
        full_measurements = []
        for _ in range(NUM_NETWORKS):
            full_measurements.extend(measurements)
        
        return np.array(full_measurements[:NUM_NETWORKS * 3], dtype=np.float32)
    
    def _get_ue_state(self, row: pd.Series) -> np.ndarray:
        """Extract UE features."""
        speed = row.get('speed_ms', 0.0)
        if pd.isna(speed):
            speed = 0.0
        return np.array([float(speed)], dtype=np.float32)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of different feature groups."""
        return self.feature_dims
    
    def get_state_size(self, state_type: str = 'classical') -> int:
        """
        Get the size of state for a given state type.
        
        Args:
            state_type: Type of state ('classical', 'lstm', 'gru')
            
        Returns:
            int: State size
        """
        if state_type in ['classical', 'lstm', 'gru']:
            return self.feature_dims['total_classical']
        return self.feature_dims['total_classical']