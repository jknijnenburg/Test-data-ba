import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

def visualize_predictions(model="RFR", country="DE", dataset="3", prediction_type="test_recalibrated"):
    """
    Quick visualization script for electricity price predictions
    
    Parameters:
    - model: Model name (RFR, SVRMulti, SVRChain)
    - country: Country code (DE, FR, BE)
    - dataset: Dataset version ("", "2", "3")
    - prediction_type: Type of predictions to load (test, test_recalibrated)
    """
    # Set the base path
    base_path = os.environ.get("EPFDAML", os.curdir)
    
    # Load predictions
    pred_path = os.path.join(
        base_path, "data", "datasets", f"EPF{dataset}_{country}", 
        f"{model}_TSCHORA_EPF{dataset}_{country}_{prediction_type}_predictions.csv"
    )
    
    # Check if the file exists
    if not os.path.exists(pred_path):
        print(f"Prediction file {pred_path} not found")
        return
    
    # Load the predictions
    predictions = pd.read_csv(pred_path, header=0)
    
    # Load test data to get actual prices
    test_path = os.path.join(base_path, "data", "datasets", f"EPF{dataset}_{country}", "test.csv")
    
    # Check if the file exists
    if not os.path.exists(test_path):
        print(f"Test file {test_path} not found")
        return
    
    # Load test data
    test_data = pd.read_csv(test_path)
    
    # Extract dates
    dates = pd.to_datetime(test_data['period_start_date'])
    
    # Extract price columns
    country_code = "DE" if country == "DE" else "FR" if country == "FR" else "BE"
    price_cols = [col for col in test_data.columns if col.startswith(f"{country_code}_price_") and
                 not col.endswith("_past_1") and not col.endswith("_past_2") and 
                 not col.endswith("_past_3") and not col.endswith("_past_7")]
    
    # Check if we have hourly data in columns
    if len(price_cols) == 24:
        print(f"Found 24 hourly price columns in test data")
        
        # Create DataFrame with actual prices (reshape to match predictions)
        actual_prices = []
        for date in dates.dt.date.unique():
            day_data = test_data[dates.dt.date == date]
            if len(day_data) > 0:
                day_prices = []
                for hour in range(24):
                    col_name = f"{country_code}_price_{hour}"
                    if col_name in test_data.columns:
                        day_prices.append(day_data[col_name].values[0])
                actual_prices.append(day_prices)
        
        actual_prices = np.array(actual_prices)
        
        # Limit to the same number of days as predictions
        if actual_prices.shape[0] > predictions.shape[0]:
            actual_prices = actual_prices[:predictions.shape[0]]
            
        # Create time index
        start_date = dates.dt.date.unique()[0]
        date_index = [start_date + timedelta(days=i) for i in range(actual_prices.shape[0])]
        
        # Plot visualizations
        plot_daily_comparisons(actual_prices, predictions.values, date_index, model, country, dataset)
        plot_weekly_comparison(actual_prices, predictions.values, date_index, model, country, dataset)
        plot_overall_metrics(actual_prices, predictions.values, date_index, model, country, dataset)
    else:
        print("Error: Could not find 24 hourly price columns in test data")

def plot_daily_comparisons(actual, predicted, dates, model, country, dataset, num_days=3):
    """
    Plot daily comparisons between actual and predicted prices
    """
    # Select a few days to visualize
    days_to_plot = min(num_days, len(dates))
    
    plt.figure(figsize=(15, 5 * days_to_plot))
    
    for i in range(days_to_plot):
        plt.subplot(days_to_plot, 1, i+1)
        
        # Get data for this day
        actual_day = actual[i]
        predicted_day = predicted[i]
        
        # Create hour ticks
        hours = range(24)
        
        # Plot
        plt.plot(hours, actual_day, 'o-', label='Actual', color='blue')
        plt.plot(hours, predicted_day, 's--', label='Predicted', color='red')
        
        # Add special hour highlighting
        for hour in [8, 9, 18, 19]:
            plt.axvspan(hour-0.5, hour+0.5, color='yellow', alpha=0.2)
            
        # Calculate metrics
        mae = np.mean(np.abs(actual_day - predicted_day))
        
        plt.title(f'Daily Comparison - {dates[i].strftime("%Y-%m-%d")} - MAE: {mae:.2f} €/MWh')
        plt.xlabel('Hour of Day')
        plt.ylabel('Price (€/MWh)')
        plt.xticks(hours)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"daily_comparison_{country}_{model}_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_weekly_comparison(actual, predicted, dates, model, country, dataset):
    """
    Plot a full week comparison
    """
    # Select first week (or as many days as available up to 7)
    days_to_plot = min(7, len(dates))
    
    plt.figure(figsize=(15, 8))
    
    # Reshape data to have hour as the x-axis
    hours = []
    actual_week = []
    predicted_week = []
    hour_labels = []
    
    for day in range(days_to_plot):
        for hour in range(24):
            hours.append(day * 24 + hour)
            actual_week.append(actual[day][hour])
            predicted_week.append(predicted[day][hour])
            if hour % 6 == 0:  # Add label every 6 hours
                hour_labels.append(f"{dates[day].strftime('%a')} {hour:02d}:00")
            else:
                hour_labels.append("")
    
    # Plot
    plt.plot(hours, actual_week, '-', label='Actual', color='blue')
    plt.plot(hours, predicted_week, '--', label='Predicted', color='red')
    
    # Highlight days
    for day in range(days_to_plot):
        plt.axvline(x=day*24, color='gray', linestyle=':', alpha=0.5)
    
    # Highlight specific hours
    for i, hour_idx in enumerate(hours):
        if i % 24 in [8, 9, 18, 19]:  # These are the hours you mentioned as special
            plt.axvspan(hour_idx-0.5, hour_idx+0.5, color='yellow', alpha=0.2)
    
    # Calculate metrics
    mae = np.mean(np.abs(np.array(actual_week) - np.array(predicted_week)))
    
    plt.title(f'Weekly Comparison - {dates[0].strftime("%Y-%m-%d")} to {dates[days_to_plot-1].strftime("%Y-%m-%d")} - MAE: {mae:.2f} €/MWh')
    plt.xlabel('Hour')
    plt.ylabel('Price (€/MWh)')
    
    # Set ticks at day boundaries
    day_ticks = [day*24 for day in range(days_to_plot+1)]
    day_labels = [dates[day].strftime('%a') if day < days_to_plot else '' for day in range(days_to_plot+1)]
    plt.xticks(day_ticks, day_labels)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"weekly_comparison_{country}_{model}_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_overall_metrics(actual, predicted, dates, model, country, dataset):
    """
    Plot overall metrics and error analysis
    """
    # Calculate daily metrics
    daily_mae = []
    daily_mape = []
    
    for day in range(len(dates)):
        daily_mae.append(np.mean(np.abs(actual[day] - predicted[day])))
        # Avoid division by zero in MAPE calculation
        mape = np.mean(np.abs((actual[day] - predicted[day]) / np.maximum(0.01, np.abs(actual[day])))) * 100
        daily_mape.append(mape)
    
    # Overall metrics
    overall_mae = np.mean(np.abs(actual.flatten() - predicted.flatten()))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Daily MAE
    ax1.bar(range(len(dates)), daily_mae, color='blue', alpha=0.7)
    ax1.set_title(f'Daily Mean Absolute Error (MAE) - Overall MAE: {overall_mae:.2f} €/MWh')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('MAE (€/MWh)')
    ax1.set_xticks(range(len(dates)))
    ax1.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates], rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Daily MAPE
    ax2.bar(range(len(dates)), daily_mape, color='green', alpha=0.7)
    ax2.set_title('Daily Mean Absolute Percentage Error (MAPE)')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_xticks(range(len(dates)))
    ax2.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"overall_metrics_{country}_{model}_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.show()

# Call the visualization function
if __name__ == "__main__":
    # You can modify these parameters as needed
    visualize_predictions(
        model="RFR",         # Model name: RFR, SVRMulti, SVRChain
        country="DE",        # Country: DE, FR, BE
        dataset="2",         # Dataset version: "", "2", "3" 
        prediction_type="test_recalibrated"  # test or test_recalibrated
    )