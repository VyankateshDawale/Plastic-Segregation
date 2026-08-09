import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_and_plot():
    print("Comparing models and generating plots...")
    
    # 1. Load results
    with open('results/classical_results.json', 'r') as f:
        classical = json.load(f)
    with open('results/deep_results.json', 'r') as f:
        deep = json.load(f)
    
    # Combined dict
    results = {}
    for k, v in classical.items():
        results[k] = {
            "test_accuracy": v["test_accuracy"],
            "test_precision": v["test_precision"],
            "test_recall": v["test_recall"],
            "test_f1": v["test_f1"],
            "confusion_matrix": v["confusion_matrix"]
        }
    for k, v in deep.items():
        results[k] = {
            "test_accuracy": v["test_accuracy"],
            "test_precision": v["test_precision"],
            "test_recall": v["test_recall"],
            "test_f1": v["test_f1"],
            "confusion_matrix": v["confusion_matrix"]
        }
        
    # Convert to DataFrame for easier handling
    df_results = pd.DataFrame(results).T
    print("\nModel Comparison Table:")
    print(df_results[["test_accuracy", "test_precision", "test_recall", "test_f1"]])
    
    # Save comparison report to CSV
    df_results.to_csv('results/model_comparison.csv')
    print("Saved comparison table to results/model_comparison.csv")
    
    # 2. Plot Model Comparison
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Reformat for plotting
    plot_data = []
    for model_name, metrics in results.items():
        plot_data.append({"Model": model_name, "Metric": "Accuracy", "Value": metrics["test_accuracy"]})
        plot_data.append({"Model": model_name, "Metric": "F1-Score", "Value": metrics["test_f1"]})
    df_plot = pd.DataFrame(plot_data)
    
    ax = sns.barplot(x="Model", y="Value", hue="Metric", data=df_plot, palette="Set2")
    plt.title("Model Performance Comparison on Test Set", pad=15)
    plt.ylabel("Score")
    plt.ylim(0.8, 1.02) # zoom in on high accuracies
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.4f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3),
                        textcoords='offset points',
                        fontsize=10)
            
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300)
    plt.close()
    print("Saved comparison plot to plots/model_comparison.png")
    
    # 3. Find best model
    best_model_name = df_results["test_f1"].idxmax()
    best_metrics = results[best_model_name]
    print(f"\nWinner: {best_model_name} with Test F1: {best_metrics['test_f1']:.4f}")
    
    # Load label mapping to display classes correctly in confusion matrix
    with open('results/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    classes = [label_mapping[str(i)] for i in range(len(label_mapping))]
    
    # 4. Save best model details
    best_info = {
        "best_model_name": best_model_name,
        "test_accuracy": best_metrics["test_accuracy"],
        "test_precision": best_metrics["test_precision"],
        "test_recall": best_metrics["test_recall"],
        "test_f1": best_metrics["test_f1"]
    }
    with open('results/best_model_info.json', 'w') as f:
        json.dump(best_info, f, indent=4)
        
    # 5. Plot confusion matrix for the best model
    cm = np.array(best_metrics["confusion_matrix"])
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(f"Confusion Matrix of the Best Model: {best_model_name}", pad=15)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig('plots/best_model_confusion_matrix.png', dpi=300)
    plt.close()
    print(f"Saved confusion matrix for {best_model_name} to plots/best_model_confusion_matrix.png")
    
    # Let's also save the confusion matrices for other models
    for model_name, metrics in results.items():
        if model_name != best_model_name:
            cm = np.array(metrics["confusion_matrix"])
            plt.figure(figsize=(8, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=classes, yticklabels=classes)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.ylabel("True Class")
            plt.xlabel("Predicted Class")
            plt.tight_layout()
            plt.savefig(f'plots/{model_name.lower()}_confusion_matrix.png', dpi=300)
            plt.close()
            
    print("All plots generated successfully!")

if __name__ == "__main__":
    evaluate_and_plot()
