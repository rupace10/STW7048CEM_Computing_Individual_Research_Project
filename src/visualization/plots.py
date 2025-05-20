"""
Visualization module for hospital readmission prediction.

This module contains functions for creating various plots and visualizations
to analyze data distributions, feature correlations, model performance,
and patient trajectories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_feature_distributions(df: pd.DataFrame, features: List[str], n_cols: int = 3) -> None:
    """
    Plot distributions of numerical features.
    
    Args:
        df (pd.DataFrame): DataFrame containing the features
        features (List[str]): List of feature names to plot
        n_cols (int): Number of columns in the subplot grid
    """
    try:
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5*n_rows))
        for i, feature in enumerate(features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=df, x=feature, kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Successfully plotted distributions for {len(features)} features")
    
    except Exception as e:
        logger.error(f"Error plotting feature distributions: {str(e)}")
        raise

def plot_correlation_matrix(df: pd.DataFrame, features: List[str], figsize: tuple = (12, 10)) -> None:
    """
    Plot correlation matrix for selected features.
    
    Args:
        df (pd.DataFrame): DataFrame containing the features
        features (List[str]): List of feature names to include
        figsize (tuple): Figure size (width, height)
    """
    try:
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True)
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        logger.info("Successfully plotted correlation matrix")
    
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {str(e)}")
        raise

def plot_feature_importance(importance_scores: Dict[str, float], top_n: int = 20) -> None:
    """
    Plot feature importance scores.
    
    Args:
        importance_scores (Dict[str, float]): Dictionary of feature names and their importance scores
        top_n (int): Number of top features to display
    """
    try:
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        features, scores = zip(*top_features)
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Successfully plotted feature importance for top {top_n} features")
    
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def plot_patient_trajectory(df: pd.DataFrame, subject_id: int, features: List[str]) -> None:
    """
    Plot patient trajectory over multiple admissions.
    
    Args:
        df (pd.DataFrame): DataFrame containing patient data
        subject_id (int): ID of the patient to plot
        features (List[str]): List of features to plot
    """
    try:
        # Filter data for the specific patient
        patient_data = df[df['subject_id'] == subject_id].sort_values('admittime')
        
        if len(patient_data) == 0:
            logger.warning(f"No data found for patient {subject_id}")
            return
        
        # Create subplots for each feature
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
        if n_features == 1:
            axes = [axes]
        
        for ax, feature in zip(axes, features):
            ax.plot(patient_data['admittime'], patient_data[feature], 'o-')
            ax.set_title(f'{feature} over time')
            ax.set_xlabel('Admission Time')
            ax.set_ylabel(feature)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Successfully plotted patient trajectory for patient {subject_id}")
    
    except Exception as e:
        logger.error(f"Error plotting patient trajectory: {str(e)}")
        raise

def plot_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> None:
    """
    Plot model performance metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_prob (Optional[np.ndarray]): Predicted probabilities
    """
    try:
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Plot ROC curve if probabilities are provided
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], 'k--')
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC Curve')
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Successfully plotted model performance metrics")
    
    except Exception as e:
        logger.error(f"Error plotting model performance: {str(e)}")
        raise 