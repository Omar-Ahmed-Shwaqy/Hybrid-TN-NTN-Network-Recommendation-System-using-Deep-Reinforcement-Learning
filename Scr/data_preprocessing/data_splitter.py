# =============================================================================
# FILE: Scr/data_preprocessing/data_splitter.py (VERSION 4.0 - PROFESSIONAL)
# =============================================================================
# PURPOSE: Advanced data splitting with multiple strategies and comprehensive
#          statistics for train/test validation.
# =============================================================================

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import USER_ID_COLUMN, NETWORK_COLUMN, AREA_COLUMN

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Advanced Data Splitter for Hybrid Network Dataset.
    
    Supports multiple splitting strategies:
    - Split by user (non-stratified)
    - Split by sequence (chronological)
    - Split by area (test on specific areas)
    - Stratified split by area or network
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        split_ratio: float = 0.8,
        random_state: int = 42,
        validate_splits: bool = True
    ):
        """
        Initialize the data splitter.
        
        Args:
            data: Full dataset
            split_ratio: Training ratio (0.0 to 1.0)
            random_state: Random seed
            validate_splits: Whether to validate splits
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.validate_splits = validate_splits
        
        self.train_data = None
        self.test_data = None
        self.split_type = None
        self.split_stats = {}
        
        # Store column info for validation
        self.user_col = USER_ID_COLUMN if USER_ID_COLUMN in data.columns else None
        self.area_col = AREA_COLUMN if AREA_COLUMN in data.columns else None
        self.network_col = NETWORK_COLUMN if NETWORK_COLUMN in data.columns else None
        
        logger.info(f"DataSplitter initialized with split_ratio={split_ratio}")
        logger.info(f"   Total samples: {len(data):,}")
        logger.info(f"   User column: {self.user_col}")
        logger.info(f"   Area column: {self.area_col}")
    
    def split_by_user(
        self, 
        random_state: Optional[int] = None,
        stratify_by: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by users.
        
        Args:
            random_state: Random seed (overrides default)
            stratify_by: Column to stratify by ('Area', 'network_type')
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.user_col is None:
            logger.warning("No user column found. Falling back to sequence split.")
            return self.split_by_sequence()
        
        random_state = random_state or self.random_state
        
        # Get unique users
        users = self.data[self.user_col].unique()
        users_list = list(users)
        n_users = len(users_list)
        
        if n_users < 2:
            logger.warning("Only 1 user found. Using sequence split.")
            return self.split_by_sequence()
        
        # Determine train size
        n_train_users = max(1, int(n_users * self.split_ratio))
        
        # Perform split
        if stratify_by and stratify_by in self.data.columns:
            # Stratified split by user attributes
            user_attributes = self.data.groupby(self.user_col)[stratify_by].first()
            user_labels = user_attributes.reindex(users_list).fillna('Unknown')
            
            train_users, test_users = train_test_split(
                users_list,
                train_size=n_train_users,
                random_state=random_state,
                stratify=user_labels
            )
            logger.info(f"Stratified split by {stratify_by}")
        else:
            # Simple split
            train_users, test_users = train_test_split(
                users_list,
                train_size=n_train_users,
                random_state=random_state
            )
            logger.info("Simple user split (non-stratified)")
        
        # Get data for each group
        self.train_data = self.data[self.data[self.user_col].isin(train_users)]
        self.test_data = self.data[self.data[self.user_col].isin(test_users)]
        self.split_type = 'user'
        
        self._compute_split_stats()
        self._validate_split()
        
        return self.train_data, self.test_data
    
    def split_by_sequence(
        self, 
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by sequence (chronological order).
        
        Args:
            shuffle: Whether to shuffle before splitting
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, test_data)
        """
        logger.info(f"Splitting by sequence (shuffle={shuffle})...")
        
        random_state = random_state or self.random_state
        
        # Sort by index for chronological order
        sorted_data = self.data.sort_index()
        
        if shuffle:
            sorted_data = sorted_data.sample(frac=1, random_state=random_state)
            logger.info("Data shuffled before split")
        
        split_idx = int(len(sorted_data) * self.split_ratio)
        
        self.train_data = sorted_data.iloc[:split_idx]
        self.test_data = sorted_data.iloc[split_idx:]
        self.split_type = 'sequence'
        
        self._compute_split_stats()
        self._validate_split()
        
        return self.train_data, self.test_data
    
    def split_by_area(
        self, 
        test_areas: Union[str, List[str]],
        balance_train: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by area (test on specific areas).
        
        Args:
            test_areas: Area(s) to use for testing
            balance_train: Whether to balance training data by area
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.area_col is None:
            raise ValueError("Area column not found in data")
        
        if isinstance(test_areas, str):
            test_areas = [test_areas]
        
        logger.info(f"Splitting by area: test_areas={test_areas}")
        
        # Split
        self.train_data = self.data[~self.data[self.area_col].isin(test_areas)]
        self.test_data = self.data[self.data[self.area_col].isin(test_areas)]
        
        # Balance training if requested
        if balance_train and len(self.train_data) > 0:
            self.train_data = self._balance_data_by_area(self.train_data)
        
        self.split_type = 'area'
        
        self._compute_split_stats()
        self._validate_split()
        
        return self.train_data, self.test_data
    
    def split_stratified(
        self,
        stratify_col: str,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified split by a column.
        
        Args:
            stratify_col: Column to stratify by
            test_size: Test set size (0.0 to 1.0)
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if stratify_col not in self.data.columns:
            raise ValueError(f"Column '{stratify_col}' not found in data")
        
        random_state = random_state or self.random_state
        
        logger.info(f"Stratified split by {stratify_col}")
        
        # Use StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - self.split_ratio,
            random_state=random_state
        )
        
        # Get indices
        X = self.data
        y = self.data[stratify_col]
        
        train_idx, test_idx = next(sss.split(X, y))
        
        self.train_data = self.data.iloc[train_idx]
        self.test_data = self.data.iloc[test_idx]
        self.split_type = 'stratified'
        
        self._compute_split_stats()
        self._validate_split()
        
        return self.train_data, self.test_data
    
    def _balance_data_by_area(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Balance data by area.
        
        Args:
            data: Input DataFrame
            
        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        if self.area_col not in data.columns:
            return data
        
        area_counts = data[self.area_col].value_counts()
        min_count = min(area_counts)
        
        if min_count > 50:
            balanced_dfs = []
            for area in area_counts.index:
                area_df = data[data[self.area_col] == area]
                if len(area_df) >= min_count:
                    sampled = area_df.sample(n=min_count, random_state=self.random_state)
                else:
                    sampled = area_df
                balanced_dfs.append(sampled)
            
            if balanced_dfs:
                return pd.concat(balanced_dfs, ignore_index=True)
        
        return data
    
    def _compute_split_stats(self) -> None:
        """Compute statistics about the split."""
        if self.train_data is None or self.test_data is None:
            return
        
        self.split_stats = {
            'split_type': self.split_type,
            'split_ratio': self.split_ratio,
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_percentage': len(self.train_data) / len(self.data) * 100,
            'test_percentage': len(self.test_data) / len(self.data) * 100,
        }
        
        # User stats
        if self.user_col and self.user_col in self.train_data.columns:
            self.split_stats['train_users'] = self.train_data[self.user_col].nunique()
            self.split_stats['test_users'] = self.test_data[self.user_col].nunique()
        
        # Area stats
        if self.area_col and self.area_col in self.train_data.columns:
            self.split_stats['train_areas'] = self.train_data[self.area_col].value_counts().to_dict()
            self.split_stats['test_areas'] = self.test_data[self.area_col].value_counts().to_dict()
    
    def _validate_split(self) -> None:
        """Validate the split for potential issues."""
        if not self.validate_splits:
            return
        
        if self.train_data is None or self.test_data is None:
            return
        
        # Check if test set is empty
        if len(self.test_data) == 0:
            logger.warning("Test set is empty!")
        
        # Check if train set is empty
        if len(self.train_data) == 0:
            logger.warning("Train set is empty!")
        
        # Check for data leakage (same user in both sets)
        if self.user_col and self.user_col in self.train_data.columns:
            train_users = set(self.train_data[self.user_col].unique())
            test_users = set(self.test_data[self.user_col].unique())
            overlap = train_users.intersection(test_users)
            if overlap:
                logger.warning(f"Data leakage detected: {len(overlap)} users in both train and test")
        
        # Check class distribution
        if self.area_col and self.area_col in self.train_data.columns:
            train_dist = self.train_data[self.area_col].value_counts(normalize=True)
            test_dist = self.test_data[self.area_col].value_counts(normalize=True)
            
            for area in train_dist.index:
                if area in test_dist.index:
                    diff = abs(train_dist[area] - test_dist[area])
                    if diff > 0.2:
                        logger.warning(f"Large distribution difference for {area}: {diff:.2f}")
    
    def get_statistics(self) -> Dict:
        """Get detailed split statistics."""
        return self.split_stats
    
    def print_summary(self) -> None:
        """Print a formatted summary of the split."""
        if self.train_data is None or self.test_data is None:
            print("❌ No split performed yet.")
            return
        
        print("\n" + "="*80)
        print("📊 DATA SPLIT SUMMARY")
        print("="*80)
        
        print(f"\n📌 Split Type: {self.split_stats.get('split_type', 'Unknown')}")
        print(f"📌 Split Ratio: {self.split_ratio:.2f}")
        print(f"📌 Random State: {self.random_state}")
        
        print(f"\n📌 Total Samples: {len(self.data):,}")
        print(f"   Train: {self.split_stats.get('train_samples', 0):,} ({self.split_stats.get('train_percentage', 0):.1f}%)")
        print(f"   Test:  {self.split_stats.get('test_samples', 0):,} ({self.split_stats.get('test_percentage', 0):.1f}%)")
        
        if 'train_users' in self.split_stats:
            print(f"\n📌 Users:")
            print(f"   Train: {self.split_stats['train_users']}")
            print(f"   Test:  {self.split_stats['test_users']}")
        
        if 'train_areas' in self.split_stats:
            print(f"\n📌 Areas (Train):")
            for area, count in self.split_stats['train_areas'].items():
                print(f"   {area}: {count:,}")
            
            print(f"\n📌 Areas (Test):")
            for area, count in self.split_stats['test_areas'].items():
                print(f"   {area}: {count:,}")
        
        print("\n" + "="*80)
    
    def save_splits(self, train_path: str, test_path: str) -> None:
        """
        Save train and test splits to files.
        
        Args:
            train_path: Path to save training data
            test_path: Path to save test data
        """
        if self.train_data is None or self.test_data is None:
            logger.error("No split performed. Call a split method first.")
            return
        
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        
        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)
        
        logger.info(f"Saved train data to: {train_path}")
        logger.info(f"Saved test data to: {test_path}")
    
    def get_train_data(self) -> pd.DataFrame:
        """Get training data."""
        return self.train_data
    
    def get_test_data(self) -> pd.DataFrame:
        """Get test data."""
        return self.test_data