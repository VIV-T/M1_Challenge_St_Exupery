import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# --- Design Configuration ---
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial', 'sans-serif']
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

PLOT_DIR = Path("exports/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_daily_momentum(daily_df, title="Project Saint-Exupéry — Daily Momentum Report"):
    """
    Time-series comparison of Daily Predicted Sum vs Daily Actual Sum.
    daily_df must have columns ['day_key', 'predicted_pax', 'NbPaxTotal']
    """
    plt.figure(figsize=(14, 7))
    
    # Sort by day
    daily_df = daily_df.sort_values('day_key')
    
    # 📈 Plotting Lines
    plt.plot(daily_df['day_key'], daily_df['NbPaxTotal'], 
             marker='o', linestyle='-', linewidth=2.5, label='Actual Ground Truth', color='#2ecc71')
    plt.plot(daily_df['day_key'], daily_df['predicted_pax'], 
             marker='s', linestyle='--', linewidth=2, label='ML Pipeline Prediction', color='#3498db')
    
    # Highlight the Surge Area
    plt.fill_between(daily_df['day_key'], daily_df['NbPaxTotal'], daily_df['predicted_pax'], 
                     color='gray', alpha=0.1)

    plt.title(title, weight='bold')
    plt.xlabel("Date (Spring 2026)", labelpad=10)
    plt.ylabel("Total Daily Passengers", labelpad=10)
    plt.legend(frameon=True, shadow=True)
    plt.xticks(rotation=45)
    
    # Save high-res asset
    save_path = PLOT_DIR / "daily_momentum.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ Momentum Asset exported to {save_path}")

def plot_feature_importance(importance_df, top_n=15):
    """
    Horizontal bar chart of LightGBM feature importances.
    importance_df must have columns ['Feature', 'Importance']
    """
    plt.figure(figsize=(10, 8))
    
    # Sort and take top_n
    data = importance_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    sns.barplot(x='Importance', y='Feature', data=data, hue='Feature', palette='viridis', legend=False)
    
    # Highlight the Breakthrough Feature
    for i, label in enumerate(data['Feature']):
        if "days_from_eid" in label.lower() or "return_surge" in label.lower():
            plt.gca().get_yticklabels()[i].set_color('red')
            plt.gca().get_yticklabels()[i].set_weight('bold')

    plt.title("Key Predictive Drivers — Project Saint-Exupéry", weight='bold')
    plt.xlabel("Relative Importance (Gain)", labelpad=10)
    plt.ylabel("Feature Manifest", labelpad=10)
    
    # Save high-res asset
    save_path = PLOT_DIR / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ Importance Asset exported to {save_path}")

def plot_error_distribution(merged_df):
    """
    Residual analysis distribution.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate Residual
    merged_df['Residual'] = merged_df['predicted_pax'] - merged_df['NbPaxTotal']
    
    sns.histplot(merged_df['Residual'], kde=True, bins=30, color='#9b59b6', alpha=0.6)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Perfect Alignment")
    
    plt.title("Model Convergence — Residual Distribution", weight='bold')
    plt.xlabel("Prediction Error (Passengers)", labelpad=10)
    plt.ylabel("Flight Frequency", labelpad=10)
    plt.legend()
    
    # Save high-res asset
    save_path = PLOT_DIR / "residual_analysis.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ Residual Asset exported to {save_path}")

def plot_hourly_distribution(hourly_df, title="Project Saint-Exupéry — Intra-Day Traffic Distribution"):
    """
    Line chart comparing Hourly Predicted Sum vs Hourly Actual Sum (Avg across period).
    hourly_df must have columns ['hour', 'predicted_pax', 'NbPaxTotal']
    """
    plt.figure(figsize=(12, 6))
    
    # Ensure hour-of-day sorting
    hourly_df = hourly_df.sort_values('hour')
    
    sns.lineplot(x='hour', y='NbPaxTotal', data=hourly_df, 
                 marker='o', label='Actual Ground Truth (Mean)', color='#e67e22', linewidth=3, alpha=0.8)
    sns.lineplot(x='hour', y='predicted_pax', data=hourly_df, 
                 marker='s', label='ML Pipeline Prediction (Mean)', color='#2c3e50', linewidth=1.5, linestyle='--')
    
    plt.fill_between(hourly_df['hour'], hourly_df['NbPaxTotal'], hourly_df['predicted_pax'], 
                     color='orange', alpha=0.1)

    plt.title(title, weight='bold')
    plt.xlabel("Hour of Day (Local Time)", labelpad=10)
    plt.ylabel("Total Passengers", labelpad=10)
    plt.xticks(range(0, 24))
    plt.legend(frameon=True)
    
    save_path = PLOT_DIR / "hourly_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ Hourly Distribution Asset exported to {save_path}")

def plot_weekly_signature(weekly_df, title="Project Saint-Exupéry — Weekly Traffic Signature"):
    """
    Bar/Line comparison of average passenger flows by Day of Week.
    Ensures Mon/Sun are always represented.
    """
    plt.figure(figsize=(10, 6))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # Ensure all days 0-6 exist for plotting
    weekly_df = weekly_df.set_index('dayofweek').reindex(range(7)).fillna(0).reset_index()
    
    plt.plot(weekly_df['dayofweek'], weekly_df['NbPaxTotal'], 
             marker='o', label='Actual Ground Truth', color='#8e44ad', linewidth=3)
    plt.plot(weekly_df['dayofweek'], weekly_df['predicted_pax'], 
             marker='s', label='ML Pipeline Prediction', color='#2c3e50', linewidth=2, linestyle='--')
    
    plt.fill_between(weekly_df['dayofweek'], weekly_df['NbPaxTotal'], weekly_df['predicted_pax'], 
                     color='purple', alpha=0.05)

    plt.title(title, weight='bold')
    plt.xlabel("Day of Week", labelpad=10)
    plt.ylabel("Average Total Passengers", labelpad=10)
    plt.xticks(range(7), days)
    plt.legend(frameon=True)
    
    save_path = PLOT_DIR / "weekly_signature.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✅ Weekly Signature Asset exported to {save_path}")
