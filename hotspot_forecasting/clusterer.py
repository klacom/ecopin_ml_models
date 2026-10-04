class DenStreamClusterer:
    def __init__(self):
        # Mocking the DenStream micro-cluster logic
        self.micro_clusters = {}
    
    def absorb(self, cluster_id: str, new_weight: float = 1.0):
        """
        Incrementally update the micro-cluster weight/density
        without retraining on historical points.
        """
        if cluster_id not in self.micro_clusters:
            self.micro_clusters[cluster_id] = {'weight': 0.0, 'reports': 0}
            
        self.micro_clusters[cluster_id]['weight'] += new_weight
        self.micro_clusters[cluster_id]['reports'] += 1
        
        print(f"[DenStream] Absorbed feedback into cluster {cluster_id}. New weight: {self.micro_clusters[cluster_id]['weight']}")
