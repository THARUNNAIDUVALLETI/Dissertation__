"""Simplified evaluation for dissertation"""
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import os

def main():
    print("="*70)
    print("HYBRID NIDS - DISSERTATION EVALUATION")
    print("="*70)
    
    # Create directories if they don't exist
    os.makedirs("../../results/metrics", exist_ok=True)
    os.makedirs("../../results/confusion_matrices", exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv("../../data/raw/UNSW-NB15_clean.csv")
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int32)
    
    # Split to recreate test set
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load scaler and transform test data
    scaler = joblib.load("../../models/saved/scaler_final.pkl")
    X_test_scaled = scaler.transform(X_test)
    
    # Load models
    print("Loading models...")
    rf_model = joblib.load("../../models/saved/rf_final.pkl")
    autoencoder = keras.models.load_model("../../models/saved/autoencoder_final.keras")
    threshold = np.load("../../models/saved/threshold_final.npy")
    
    # Get predictions
    print("Generating predictions...")
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    reconstructions = autoencoder.predict(X_test_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
    y_pred_hybrid = np.logical_or(y_pred_rf == 1, reconstruction_errors > threshold).astype(int)
    
    # Calculate confusion matrices
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_hybrid = confusion_matrix(y_test, y_pred_hybrid)
    
    # Extract values
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    tn_hy, fp_hy, fn_hy, tp_hy = cm_hybrid.ravel()
    
    # Calculate metrics for Random Forest
    rf_metrics = {
        'model': 'Random Forest',
        'accuracy': (tp_rf + tn_rf) / (tp_rf + tn_rf + fp_rf + fn_rf),
        'precision': tp_rf / (tp_rf + fp_rf) if (tp_rf + fp_rf) > 0 else 0,
        'recall': tp_rf / (tp_rf + fn_rf) if (tp_rf + fn_rf) > 0 else 0,
        'f1_score': 2 * tp_rf / (2 * tp_rf + fp_rf + fn_rf) if (2 * tp_rf + fp_rf + fn_rf) > 0 else 0,
        'false_positive_rate': fp_rf / (fp_rf + tn_rf) if (fp_rf + tn_rf) > 0 else 0,
        'false_negative_rate': fn_rf / (fn_rf + tp_rf) if (fn_rf + tp_rf) > 0 else 0,
    }
    
    # Calculate metrics for Hybrid model
    hybrid_metrics = {
        'model': 'Hybrid (RF+Autoencoder)',
        'accuracy': (tp_hy + tn_hy) / (tp_hy + tn_hy + fp_hy + fn_hy),
        'precision': tp_hy / (tp_hy + fp_hy) if (tp_hy + fp_hy) > 0 else 0,
        'recall': tp_hy / (tp_hy + fn_hy) if (tp_hy + fn_hy) > 0 else 0,
        'f1_score': 2 * tp_hy / (2 * tp_hy + fp_hy + fn_hy) if (2 * tp_hy + fp_hy + fn_hy) > 0 else 0,
        'false_positive_rate': fp_hy / (fp_hy + tn_hy) if (fp_hy + tn_hy) > 0 else 0,
        'false_negative_rate': fn_hy / (fn_hy + tp_hy) if (fn_hy + tp_hy) > 0 else 0,
    }
    
    # Display results
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    print("\nRandom Forest Results:")
    print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"  Precision: {rf_metrics['precision']:.4f}")
    print(f"  Recall: {rf_metrics['recall']:.4f}")
    print(f"  F1-Score: {rf_metrics['f1_score']:.4f}")
    print(f"  False Positive Rate: {rf_metrics['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {rf_metrics['false_negative_rate']:.4f}")
    
    print("\nHybrid Model Results:")
    print(f"  Accuracy: {hybrid_metrics['accuracy']:.4f}")
    print(f"  Precision: {hybrid_metrics['precision']:.4f}")
    print(f"  Recall: {hybrid_metrics['recall']:.4f}")
    print(f"  F1-Score: {hybrid_metrics['f1_score']:.4f}")
    print(f"  False Positive Rate: {hybrid_metrics['false_positive_rate']:.4f}")
    print(f"  False Negative Rate: {hybrid_metrics['false_negative_rate']:.4f}")
    
    # Confusion matrices
    print("\n" + "="*70)
    print("CONFUSION MATRICES")
    print("="*70)
    
    print("\nRandom Forest Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Benign  Malicious")
    print(f"  Actual Benign   {tn_rf:4d}     {fp_rf:4d}")
    print(f"         Malicious {fn_rf:4d}     {tp_rf:4d}")
    
    print("\nHybrid Model Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Benign  Malicious")
    print(f"  Actual Benign   {tn_hy:4d}     {fp_hy:4d}")
    print(f"         Malicious {fn_hy:4d}     {tp_hy:4d}")
    
    # Save to CSV
    comparison_df = pd.DataFrame([rf_metrics, hybrid_metrics])
    comparison_df.to_csv("../../results/metrics/comparison.csv", index=False)
    print("\n Results saved to results/metrics/comparison.csv")
    
    # Hypothesis testing
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING")
    print("="*70)
    
    fpr_improvement = ((rf_metrics['false_positive_rate'] - hybrid_metrics['false_positive_rate']) 
                       / rf_metrics['false_positive_rate'] * 100) if rf_metrics['false_positive_rate'] > 0 else 0
    
    print(f"\nRandom Forest FPR: {rf_metrics['false_positive_rate']:.4f}")
    print(f"Hybrid Model FPR: {hybrid_metrics['false_positive_rate']:.4f}")
    print(f"FPR Reduction: {fpr_improvement:.1f}%")
    
    if fpr_improvement >= 30:
        print("\n✓ HYPOTHESIS SUPPORTED: 30% FPR reduction achieved")
    else:
        print(f"\nNote: {fpr_improvement:.1f}% reduction (target was 30%)")
    
    # Classification reports
    print("\n" + "="*70)
    print("CLASSIFICATION REPORTS")
    print("="*70)
    
    print("\nRandom Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=['Benign', 'Malicious']))
    
    print("\nHybrid Model:")
    print(classification_report(y_test, y_pred_hybrid, target_names=['Benign', 'Malicious']))
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
