"""
Testing (Verification and Validation) Script for the AI-Powered Customer Insight Tool

Purpose:
This script serves as the formal testing and validation component for the project. 
Its sole function is to run the model parameter validation to determine the optimal 
number of clusters (k) for the K-Means algorithm.

It achieves this by importing the necessary data preparation and evaluation functions 
from the main application script (`customer_segmentation_v2.py`) and executing only 
the validation steps.

This modular approach separates model validation from the full data processing pipeline,
adhering to good software design principles.

Outputs:
- A plot saved as `k_evaluation_metrics.png` showing the Elbow Method and Silhouette Scores.
- Console output indicating the optimal 'k' suggested by the Silhouette Score.
"""

# Import the specific functions needed for validation from the main script
from customer_segmentation_v2 import prepare_rfm_data, preprocess_and_scale, evaluate_optimal_k

# --- Configuration ---
# This must point to the same data file used by the main script.
DATA_FILE = 'online_retail.csv'

# --- Main Validation Execution ---
def main():
    """
    Main function to execute the validation pipeline.
    """
    print("=" * 60)
    print("RUNNING MODEL VALIDATION SCRIPT")
    print("=" * 60)

    try:
        # Step 1: Prepare the data for the model (same as the main script)
        # We need to process the data to get the features that the model uses.
        print("\n[Validation Step 1/3] Preparing and scaling data...")
        rfm_raw = prepare_rfm_data(DATA_FILE)
        rfm_scaled = preprocess_and_scale(rfm_raw.copy())
        print("Data successfully prepared for validation.")

        # Step 2: Run the model evaluation (The Core Test)
        # This function performs the actual validation by testing k=2 through 10.
        print("\n[Validation Step 2/3] Evaluating optimal K using Elbow and Silhouette Score...")
        best_k = evaluate_optimal_k(rfm_scaled, max_k=10)
        
        # Step 3: Report the results
        print("\n[Validation Step 3/3] Validation complete.")
        print("-" * 30)
        print(f"Optimal number of clusters (k) based on Silhouette Score: {best_k}")
        print(f"Validation chart has been saved as 'k_evaluation_metrics.png'")
        print("-" * 30)

    except FileNotFoundError:
        print(f"\n[ERROR] The data file '{DATA_FILE}' was not found.")
        print("Please make sure the CSV file is in the same directory as the script.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

    print("\n" + "=" * 60)
    print("VALIDATION SCRIPT FINISHED")
    print("=" * 60)

if __name__ == "__main__":
    main()
