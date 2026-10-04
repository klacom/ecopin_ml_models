from forecaster import TrendForecaster, LSTMEscalationPredictor
from clusterer import DenStreamClusterer

# Initialize singleton models
trend_forecaster = TrendForecaster()
lstm_predictor = LSTMEscalationPredictor()
cluster_manager = DenStreamClusterer()

def handle_feedback(payload: dict):
    """
    Step 5 of the pipeline lifecycle: Update
    Triggered via the POST /feedback webhook.
    """
    cluster_id = payload.get('cluster_id')
    outcome = payload.get('outcome', 0)
    
    # Extract features for ML
    crew_count = payload.get('crew_count', 1)
    overtime_ratio = payload.get('overtime_ratio', 1.0)
    
    # 1. Update SGD Trend Forecaster
    features = [crew_count, overtime_ratio]
    trend_forecaster.partial_fit(features, outcome)
    
    # 2. Update LSTM / Sequential Predictor
    if cluster_id:
        lstm_predictor.update_sequence(cluster_id, features, outcome)
        
        # 3. Update spatial micro-cluster weight
        # Higher overtime -> higher weight/severity in spatial density
        weight_increase = 1.0 + (overtime_ratio - 1.0)
        cluster_manager.absorb(cluster_id, new_weight=weight_increase)
        
    print("[TrainingLoop] Feedback absorbed successfully. Model state updated in-place.")
