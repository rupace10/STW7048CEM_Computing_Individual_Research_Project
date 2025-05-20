"""
Feature engineering module for hospital readmission prediction.

This module contains functions for creating and transforming features
from the hospital data, including temporal features, clinical features,
and patient history features.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from datetime columns.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime columns
        
    Returns:
        pd.DataFrame: DataFrame with added temporal features
    """
    try:
        df = df.copy()
        
        # Extract temporal features from admittime
        df['admit_dayofweek'] = df['admittime'].dt.dayofweek
        df['admit_month'] = df['admittime'].dt.month
        df['admit_season'] = df['admittime'].dt.month % 12 // 3 + 1
        df['admit_year'] = df['admittime'].dt.year
        
        # Create holiday flag (simplified version)
        df['is_holiday'] = df['admit_dayofweek'].isin([5, 6]).astype(int)  # Weekend as holiday
        
        logger.info("Successfully created temporal features")
        return df
    
    except Exception as e:
        logger.error(f"Error creating temporal features: {str(e)}")
        raise

def create_patient_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on patient history.
    
    Args:
        df (pd.DataFrame): DataFrame with patient admission data
        
    Returns:
        pd.DataFrame: DataFrame with added patient history features
    """
    try:
        df = df.copy()
        
        # Sort by patient and admission time
        df = df.sort_values(['subject_id', 'admittime'])
        
        # Number of previous admissions
        df['previous_admissions_count'] = df.groupby('subject_id').cumcount()
        
        # Previous length of stay
        df['previous_los'] = df.groupby('subject_id')['los'].shift(1)
        
        # Time since last discharge
        df['previous_dischtime'] = df.groupby('subject_id')['dischtime'].shift(1)
        df['time_since_last_discharge'] = (df['admittime'] - df['previous_dischtime']).dt.total_seconds() / (60*60*24)
        
        # Fill missing values for first admissions
        df['previous_los'].fillna(0, inplace=True)
        df['time_since_last_discharge'].fillna(0, inplace=True)
        
        logger.info("Successfully created patient history features")
        return df
    
    except Exception as e:
        logger.error(f"Error creating patient history features: {str(e)}")
        raise

def create_clinical_features(df: pd.DataFrame, diagnoses_df: pd.DataFrame, procedures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create clinical features from diagnoses and procedures.
    
    Args:
        df (pd.DataFrame): Main DataFrame
        diagnoses_df (pd.DataFrame): Diagnoses DataFrame
        procedures_df (pd.DataFrame): Procedures DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added clinical features
    """
    try:
        # Count diagnoses per admission
        diagnoses_count = diagnoses_df.groupby('hadm_id').size().reset_index(name='num_diagnoses')
        
        # Count procedures per admission
        procedures_count = procedures_df.groupby('hadm_id').size().reset_index(name='num_procedures')
        
        # Merge with main DataFrame
        df = df.merge(diagnoses_count, on='hadm_id', how='left')
        df = df.merge(procedures_count, on='hadm_id', how='left')
        
        # Fill missing values
        df['num_diagnoses'].fillna(0, inplace=True)
        df['num_procedures'].fillna(0, inplace=True)
        
        logger.info("Successfully created clinical features")
        return df
    
    except Exception as e:
        logger.error(f"Error creating clinical features: {str(e)}")
        raise

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different variables.
    
    Args:
        df (pd.DataFrame): DataFrame with basic features
        
    Returns:
        pd.DataFrame: DataFrame with added interaction features
    """
    try:
        df = df.copy()
        
        # Age × Charlson score interaction
        if 'anchor_age' in df.columns and 'charlson_index' in df.columns:
            df['age_charlson_interaction'] = df['anchor_age'] * df['charlson_index']
        
        # Length of stay × number of diagnoses interaction
        if 'los' in df.columns and 'num_diagnoses' in df.columns:
            df['los_diagnoses_interaction'] = df['los'] * df['num_diagnoses']
        
        # Previous admissions × current length of stay interaction
        if 'previous_admissions_count' in df.columns and 'los' in df.columns:
            df['prev_adm_los_interaction'] = df['previous_admissions_count'] * df['los']
        
        logger.info("Successfully created interaction features")
        return df
    
    except Exception as e:
        logger.error(f"Error creating interaction features: {str(e)}")
        raise 