"""
Data preprocessing module for hospital readmission prediction.

This module contains functions for loading and preprocessing hospital data,
including data cleaning, handling missing values, and basic transformations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load data from CSV files into pandas DataFrames.
    
    Args:
        data_paths (Dict[str, str]): Dictionary mapping dataset names to file paths
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing loaded DataFrames
        
    Raises:
        FileNotFoundError: If any of the specified files cannot be found
        pd.errors.EmptyDataError: If any of the files are empty
    """
    data = {}
    for name, path in data_paths.items():
        try:
            data[name] = pd.read_csv(path)
            logger.info(f"Successfully loaded {name} from {path}")
        except FileNotFoundError:
            logger.error(f"Error: {name} file not found at {path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Error: {name} file is empty at {path}")
            raise
    return data

def clean_admissions_data(admissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess admissions data.
    
    Args:
        admissions_df (pd.DataFrame): Raw admissions DataFrame
        
    Returns:
        pd.DataFrame: Cleaned admissions DataFrame
    """
    try:
        # Create a copy to avoid modifying the original
        admissions = admissions_df.copy()
        
        # Convert datetime columns
        datetime_columns = ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']
        for col in datetime_columns:
            if col in admissions.columns:
                admissions[col] = pd.to_datetime(admissions[col], errors='coerce')
        
        # Handle missing values
        admissions['discharge_location'] = admissions['discharge_location'].fillna('UNKNOWN')
        admissions['insurance'] = admissions['insurance'].fillna('UNKNOWN')
        admissions['language'] = admissions['language'].fillna('UNKNOWN')
        admissions['marital_status'] = admissions['marital_status'].fillna('UNKNOWN')
        
        logger.info("Successfully cleaned admissions data")
        return admissions
    
    except Exception as e:
        logger.error(f"Error cleaning admissions data: {str(e)}")
        raise

def merge_patient_data(admissions_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge admissions and patient data.
    
    Args:
        admissions_df (pd.DataFrame): Admissions DataFrame
        patients_df (pd.DataFrame): Patients DataFrame
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    try:
        merged_df = pd.merge(admissions_df, patients_df, on='subject_id', how='left')
        logger.info("Successfully merged admissions and patient data")
        return merged_df
    
    except Exception as e:
        logger.error(f"Error merging patient data: {str(e)}")
        raise

def calculate_length_of_stay(admissions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate length of stay for each admission.
    
    Args:
        admissions_df (pd.DataFrame): Admissions DataFrame with datetime columns
        
    Returns:
        pd.DataFrame: DataFrame with added length of stay column
    """
    try:
        admissions = admissions_df.copy()
        admissions['los'] = (admissions['dischtime'] - admissions['admittime']).dt.total_seconds() / (60*60*24)
        
        # Handle negative or unreasonable values
        admissions.loc[admissions['los'] < 0, 'los'] = 0
        admissions.loc[admissions['los'] > 365, 'los'] = 365  # Cap at 1 year
        
        logger.info("Successfully calculated length of stay")
        return admissions
    
    except Exception as e:
        logger.error(f"Error calculating length of stay: {str(e)}")
        raise 