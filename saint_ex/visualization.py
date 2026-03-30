"""
visualization.py — Save plots for analysis and reporting.

Generates:
  - Feature importance (top 30)
  - Daily predictions vs actuals
  - Residual distribution
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .config import OUTPUT_DIR


def save_plots(lgb_model, df_val, y_val, y_pred):
    """Save all plots to the output directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    _plot_feature_importance(lgb_model)
    _plot_daily_predictions(df_val, y_val, y_pred)
    _plot_residuals(y_val, y_pred)

    print("  Plots saved.")


def _plot_feature_importance(model):
    """Bar chart of the 30 most important features."""
    fi = pd.DataFrame({
        'Feature':    model.feature_name_,
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=False).head(30)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=fi, x='Importance', y='Feature', palette='Blues_r', ax=ax)
    ax.set_title('Top 30 Features (LightGBM)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150)
    plt.close()


def _plot_daily_predictions(df_val, y_val, y_pred):
    """Time series: actual vs predicted daily passenger totals."""
    daily = pd.DataFrame({
        'date':   pd.to_datetime(df_val['LTScheduledDatetime'].values).date,
        'actual': y_val.values,
        'pred':   y_pred,
    }).groupby('date')[['actual', 'pred']].sum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily['actual'], label='Actual', alpha=0.8, linewidth=1.5)
    ax.plot(daily.index, daily['pred'],   label='Predicted', alpha=0.8, linewidth=1.5)
    ax.set_title('Daily Passengers — Validation')
    ax.set_ylabel('Total Passengers')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'daily_predictions.png', dpi=150)
    plt.close()


def _plot_residuals(y_val, y_pred):
    """Histogram of prediction errors."""
    residuals = y_pred - y_val.values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title('Residuals (predicted − actual)')
    ax.set_xlabel('Error (passengers)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'residuals.png', dpi=150)
    plt.close()
