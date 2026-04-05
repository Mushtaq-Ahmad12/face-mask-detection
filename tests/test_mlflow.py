import os
import mlflow

def test_mlflow_import():
    """Verifies that the required MLflow pipelines are accessible."""
    import mlflow.tensorflow
    assert mlflow is not None
    assert mlflow.tensorflow is not None
    print("[SUCCESS] MLflow and MLflow.TensorFlow libraries imported correctly!")

def test_mlflow_connection():
    """
    Submits a dummy ping to your DagsHub or local MLflow tracking server 
    to verify network credentials and connection variables are working.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "Local Default")
    print(f"\nTarget Tracking URI: {tracking_uri}")
    
    try:
        # Create or target a dummy diagnostic experiment
        mlflow.set_experiment("diagnostic_ping_test")
        
        with mlflow.start_run(run_name="backend_connection_test"):
            # Ping the backend with random data
            mlflow.log_param("dagshub_integration_active", True)
            mlflow.log_metric("diagnostic_ping", 100.0)
            
        print("[SUCCESS] Connected and successfully wrote parameters to the MLflow tracking server!")
    except Exception as e:
        print(f"[ERROR] Failed to communicate with the MLflow backend.\nReason: {e}")
        print("\nIf you are using DagsHub, ensure you have set MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, and MLFLOW_TRACKING_PASSWORD in your environment!")

if __name__ == "__main__":
    print("--- Starting MLflow Integration Diagnostics ---")
    test_mlflow_import()
    test_mlflow_connection()
    print("---------------------------------------------")
