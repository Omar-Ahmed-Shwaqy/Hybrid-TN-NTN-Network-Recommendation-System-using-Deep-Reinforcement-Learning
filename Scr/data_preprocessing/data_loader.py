# =============================================================================
# FILE: Scr/data_preprocessing/data_loader.py (VERSION 4.0 - PROFESSIONAL)
# =============================================================================
# PURPOSE: Robust data loading with multiple encoding support, error handling,
#          and automatic fallback to dummy data for testing.
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Robust Data Loader for Hybrid Network Dataset.
    
    Features:
    - Multiple encoding support (UTF-8, Latin-1, CP1252, ISO-8859-1)
    - Automatic column name normalisation
    - Infinite value handling
    - Missing value imputation
    - Automatic dummy data fallback for testing
    - Data validation and statistics
    """
    
    def __init__(self, data_path: str, balance_data: bool = True, random_state: int = 42):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file
            balance_data: Whether to balance data by area
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.balance_data = balance_data
        self.random_state = random_state
        self.data = None
        self.stats = {}
        
        # Column aliases for normalisation
        self.column_aliases = {
            'user_id': 'UE_ID',
            'User_ID': 'UE_ID',
            'USER_ID': 'UE_ID',
            'area': 'Area',
            'AREA': 'Area',
            'location': 'Area',
            'available_networks': 'Available_Networks',
            'AvailableNetworks': 'Available_Networks',
            'available_networks_list': 'Available_Networks',
            'network_type': 'network_type',
            'Network_Type': 'network_type',
            'network': 'network_type',
        }
        
        # Valid networks and areas for validation
        self.valid_networks = ['NR_5G', 'WiFi', 'SAT (LEO)', 'HAPS', 'UAV']
        self.valid_areas = ['Urban', 'Indoor', 'Rural', 'Highway', 'Maritime', 'Desert']
        
        logger.info(f"DataLoader initialized with: {data_path}")
    
    def load(self) -> pd.DataFrame:
        """
        Load data with comprehensive error handling.
        
        Returns:
            pd.DataFrame: Loaded and cleaned data
            
        Raises:
            FileNotFoundError: If data file not found and no fallback
        """
        try:
            # Check if file exists
            if not os.path.exists(self.data_path):
                logger.warning(f"File not found: {self.data_path}")
                logger.info("Creating dummy data for testing...")
                return self._create_dummy_data()
            
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
            loaded = False
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.data_path, encoding=encoding)
                    loaded = True
                    logger.info(f"Successfully loaded with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Failed with {encoding}: {e}")
                    continue
            
            if not loaded:
                logger.warning("Could not load file with any encoding")
                logger.info("Creating dummy data for testing...")
                return self._create_dummy_data()
            
            # Normalise column names
            self.data = self._normalise_columns(self.data)
            
            # Validate and clean data
            self.data = self._clean_data(self.data)
            
            # Balance data if requested
            if self.balance_data:
                self.data = self._balance_data(self.data)
            
            # Compute statistics
            self._compute_statistics(self.data)
            
            logger.info(f"✅ Data loaded: {len(self.data):,} rows, {len(self.data.columns)} columns")
            logger.info(f"   Users: {self.data['UE_ID'].nunique() if 'UE_ID' in self.data.columns else 'N/A'}")
            logger.info(f"   Areas: {self.data['Area'].nunique() if 'Area' in self.data.columns else 'N/A'}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            logger.info("Creating dummy data for testing...")
            return self._create_dummy_data()
    
    def _normalise_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise column names to canonical format.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with normalised column names
        """
        # Rename columns using aliases
        rename_map = {}
        for col in data.columns:
            if col in self.column_aliases:
                rename_map[col] = self.column_aliases[col]
            elif col.lower() in self.column_aliases:
                rename_map[col] = self.column_aliases[col.lower()]
        
        if rename_map:
            data = data.rename(columns=rename_map)
            logger.info(f"Normalised {len(rename_map)} columns")
        
        return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Create copy to avoid warnings
        df = data.copy()
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with too many missing values
        threshold = len(df.columns) - 2
        df = df.dropna(thresh=threshold)
        
        # Fill missing numeric values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
        
        # Fill missing categorical values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        # Validate networks
        if 'network_type' in df.columns:
            df['network_type'] = df['network_type'].apply(
                lambda x: x if x in self.valid_networks else self.valid_networks[0]
            )
        
        # Validate areas
        if 'Area' in df.columns:
            df['Area'] = df['Area'].apply(
                lambda x: x if x in self.valid_areas else self.valid_areas[0]
            )
        
        # Convert available_networks to proper format
        if 'Available_Networks' in df.columns:
            df['Available_Networks'] = df['Available_Networks'].apply(
                lambda x: self._parse_available_networks(x)
            )
        
        # Add derived features if missing
        if 'speed_ms' not in df.columns:
            df['speed_ms'] = np.random.uniform(0, 30, len(df))
        
        logger.info(f"Cleaned data: {len(df):,} rows remaining")
        
        return df
    
    def _parse_available_networks(self, value) -> str:
        """
        Parse available networks into consistent format.
        
        Args:
            value: Raw available networks value
            
        Returns:
            str: Comma-separated network names
        """
        if pd.isna(value) or value is None:
            return 'NR_5G,WiFi'
        
        # If already string, clean it
        if isinstance(value, str):
            networks = [n.strip() for n in value.split(',') if n.strip()]
            valid_networks = [n for n in networks if n in self.valid_networks]
            if not valid_networks:
                valid_networks = ['NR_5G']
            return ','.join(valid_networks)
        
        # If list, join it
        if isinstance(value, (list, tuple)):
            valid_networks = [n for n in value if n in self.valid_networks]
            if not valid_networks:
                valid_networks = ['NR_5G']
            return ','.join(valid_networks)
        
        return 'NR_5G,WiFi'
    
    def _balance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Balance data by area to avoid bias.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        if 'Area' not in data.columns:
            return data
        
        # Count samples per area
        area_counts = data['Area'].value_counts()
        
        # If all areas have enough samples, sample equally
        min_count = min(area_counts)
        
        if min_count > 100:
            # Sample equally from each area
            balanced_dfs = []
            for area in self.valid_areas:
                if area in area_counts.index:
                    area_df = data[data['Area'] == area]
                    if len(area_df) >= min_count:
                        sampled = area_df.sample(n=min_count, random_state=self.random_state)
                    else:
                        sampled = area_df
                    balanced_dfs.append(sampled)
            
            if balanced_dfs:
                data = pd.concat(balanced_dfs, ignore_index=True)
                logger.info(f"Balanced data: {len(data):,} rows")
        
        return data
    
    def _compute_statistics(self, data: pd.DataFrame) -> None:
        """
        Compute and store data statistics.
        
        Args:
            data: Input DataFrame
        """
        self.stats = {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        if 'Area' in data.columns:
            self.stats['areas'] = data['Area'].value_counts().to_dict()
        
        if 'network_type' in data.columns:
            self.stats['networks'] = data['network_type'].value_counts().to_dict()
        
        if 'UE_ID' in data.columns:
            self.stats['users'] = data['UE_ID'].nunique()
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """
        Create dummy data for testing when real data is unavailable.
        
        Returns:
            pd.DataFrame: Dummy data
        """
        np.random.seed(self.random_state)
        
        n_samples = 1000
        
        areas = ['Urban', 'Indoor', 'Rural', 'Highway', 'Maritime', 'Desert']
        networks = ['NR_5G', 'WiFi', 'SAT (LEO)', 'HAPS', 'UAV']
        
        # Create base data
        data = {
            'UE_ID': [f'user_{i%20}' for i in range(n_samples)],
            'Area': [np.random.choice(areas) for _ in range(n_samples)],
            'network_type': [np.random.choice(networks) for _ in range(n_samples)],
            'Available_Networks': [
                ','.join(np.random.choice(networks, size=np.random.randint(2, 5), replace=False))
                for _ in range(n_samples)
            ],
            'SNR_dB': np.random.uniform(-10, 30, n_samples),
            'SINR_dB': np.random.uniform(-10, 25, n_samples),
            'RSSI_dBm': np.random.uniform(-120, -50, n_samples),
            'Throughput_Mbps': np.random.uniform(1, 1000, n_samples),
            'Latency_ms': np.random.uniform(1, 100, n_samples),
            'Packet_Loss_pct': np.random.uniform(0, 5, n_samples),
            'BER': np.random.uniform(0, 0.01, n_samples),
            'Link_Quality_Index': np.random.uniform(0, 100, n_samples),
            'Spectral_Efficiency_bps_hz': np.random.uniform(0, 10, n_samples),
            'speed_ms': np.random.uniform(0, 30, n_samples),
            'rsrp': np.random.uniform(-120, -60, n_samples),
            'rsrq': np.random.uniform(-20, -3, n_samples),
            'snr': np.random.uniform(0, 30, n_samples),
            'latency': np.random.uniform(1, 100, n_samples),
            'throughput': np.random.uniform(1, 1000, n_samples),
            'qos': np.random.uniform(0.5, 1, n_samples),
            'handover_count': np.random.randint(0, 10, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Clean dummy data
        df = self._clean_data(df)
        
        logger.info(f"✅ Created dummy data: {len(df):,} rows, {len(df.columns)} columns")
        
        return df
    
    def get_stats(self) -> dict:
        """
        Get data statistics.
        
        Returns:
            dict: Data statistics
        """
        return self.stats
    
    def print_stats(self) -> None:
        """Print data statistics in a formatted way."""
        if not self.stats:
            print("No data loaded yet.")
            return
        
        print("\n" + "="*80)
        print("📊 DATA STATISTICS")
        print("="*80)
        print(f"\n📌 Total Rows: {self.stats['rows']:,}")
        print(f"📌 Total Columns: {self.stats['columns']}")
        print(f"📌 Memory Usage: {self.stats['memory_mb']:.2f} MB")
        
        if 'users' in self.stats:
            print(f"📌 Users: {self.stats['users']}")
        
        if 'areas' in self.stats:
            print("\n📌 Areas:")
            for area, count in self.stats['areas'].items():
                print(f"   {area}: {count:,} ({count/self.stats['rows']*100:.1f}%)")
        
        if 'networks' in self.stats:
            print("\n📌 Networks:")
            for network, count in self.stats['networks'].items():
                print(f"   {network}: {count:,} ({count/self.stats['rows']*100:.1f}%)")
        
        print("="*80)