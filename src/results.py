import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.models import train_model


# Function to view model performances for a specific metric ['mse', 'r2', 'rmse']
def plot_model_performance(results, metric='mse'):
    summary_data = []
    for model_type, res in results.items():
        train_mean = np.mean(res['train_' + metric])
        train_std = np.std(res['train_' + metric])
        test_mean = np.mean(res['test_' + metric])
        test_std = np.std(res['test_' + metric])

        summary_data.append({
            'Model': model_type,
            'Dataset': 'Train',
            'Mean_' + metric.upper(): train_mean,
            'Std_' + metric.upper(): train_std
        })
        summary_data.append({
            'Model': model_type,
            'Dataset': 'Test',
            'Mean_' + metric.upper(): test_mean,
            'Std_' + metric.upper(): test_std
        })

    summary_df = pd.DataFrame(summary_data)

    plt.figure(figsize=(14,8))
    sns.barplot(x='Model', y='Mean_' + metric.upper(), hue='Dataset', data=summary_df, errorbar=None)
    
    plt.title(f'Model Performance Comparison (Mean {metric.upper()})')
    plt.ylabel(f'Mean Squared Error ({metric.upper()})')
    plt.xlabel('Model')
    plt.show()

    return summary_df


# Function to plot residuals and scatter plot for each fold
def plot_residuals_and_scatter(y_true_fold, y_pred_fold, model_type, fold = None, dataset='Test'):
    residuals = y_true_fold - y_pred_fold
    
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    
    # Plot residuals distribution
    sns.histplot(residuals, kde=True, bins=30, ax=axs[0])
    if fold != None:
        axs[0].set_title(f'Residuals Distribution for {model_type} (Fold {fold+1}, {dataset} Set)')
    else:
        axs[0].set_title(f'Residuals Distribution for {model_type}, {dataset} Set)')
    axs[0].set_xlabel('Residuals')
    axs[0].set_ylabel('Density')
    axs[0].axvline(0, color='red', linestyle='--')
    axs[0].text(0.95,0.95, 
                f'Mean: {mean_residuals:.2f}\nStd: {std_residuals:.2f}',
                transform=axs[0].transAxes, fontsize=12, 
                ha='right', va='top', 
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Plot scatter plot
    axs[1].scatter(y_true_fold, y_pred_fold, alpha=0.5)
    axs[1].plot([y_true_fold.min(), y_true_fold.max()], 
                [y_true_fold.min(), y_true_fold.max()], 'r--')
    axs[1].set_title('True vs Prediction')
    axs[1].set_xlabel('True Values')
    axs[1].set_ylabel('Predicted Values')
    
    plt.show()


# Function to plot coefficients vs alpha values
def plot_coefficients_vs_alpha(X_train, y_train, alpha_values, model_types=['ridge', 'lasso', 'elasticnet'], l1_ratio=0.5):
    coefficients_dict = {model_type: [] for model_type in model_types}

    for alpha in alpha_values:
        for model_type in model_types:
            model = train_model(X_train, y_train, model_type=model_type, alpha=alpha, l1_ratio=l1_ratio)
            coefficients_dict[model_type].append(model.coef_)

    print("-"*30)

    for model_type in model_types:
        coefficients_dict[model_type] = np.array(coefficients_dict[model_type])

    for index, alpha in enumerate(alpha_values):
        fig = plt.figure(figsize=(10,6))
        for model_type in model_types:
            plt.plot(range(X_train.shape[1]), coefficients_dict[model_type][index,:], marker = 'o', label = model_type)

        ymin, ymax = plt.ylim()
        yticks = np.arange(np.floor(ymin / 1e6) * 1e6, np.ceil(ymax / 1e6) * 1e6 + 0.3*1e6, 0.3*1e6)
        plt.yticks(yticks)
        
        plt.xlabel('Coefficient Index')
        plt.ylabel('Coefficient Value')
        plt.title(f'Coefficient Values for Alpha = {alpha}')
        plt.legend(loc='upper right')
        plt.show()
