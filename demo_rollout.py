import torch
import numpy as np
from model.optimized_mscred import OptimizedMSCRED

def demo_temporal_rollout():
    """
    Démontre l'usage du rollout temporel optimisé
    """
    
    # Initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedMSCRED(in_channels_encoder=3, in_channels_decoder=256, sequence_length=5)
    model.to(device)
    model.eval()
    
    # Simulation de données temporelles
    batch_size = 4
    channels = 3
    height, width = 30, 30  # Taille des matrices de signature
    
    print("=== STRATÉGIE 1: Traitement frame par frame (streaming) ===")
    
    # Reset buffers pour nouvelle séquence
    model.reset_buffers()
    
    # Simuler une séquence temporelle de 10 frames
    for t in range(10):
        # Nouvelle frame (batch de matrices de signature)
        x_new = torch.randn(batch_size, channels, height, width).to(device)
        
        print(f"Frame {t+1}:")
        
        with torch.no_grad():
            prediction = model(x_new, return_prediction=True)
            
            if prediction is not None:
                print(f"  ✓ Prédiction générée: {prediction.shape}")
                # Ici vous pourriez calculer la loss, sauvegarder, etc.
            else:
                print(f"  ○ Buffer en cours de remplissage ({len(model.feature_buffers['conv1'])}/5)")
    
    print("\n=== STRATÉGIE 2: Traitement par batch complet ===")
    
    # Pour l'entraînement, on peut traiter des séquences complètes
    sequence_length = 8
    full_sequence = torch.randn(batch_size, sequence_length, channels, height, width).to(device)
    
    model.reset_buffers()
    predictions = []
    
    with torch.no_grad():
        for t in range(sequence_length):
            x_t = full_sequence[:, t]  # Frame au temps t
            pred = model(x_t, return_prediction=True)
            
            if pred is not None:
                predictions.append(pred)
    
    if predictions:
        predictions_tensor = torch.stack(predictions, dim=1)  # [batch, time, channels, h, w]
        print(f"Séquence de prédictions: {predictions_tensor.shape}")
    
    print("\n=== STRATÉGIE 3: Entraînement avec fenêtre glissante ===")
    
    def train_step_windowed(model, optimizer, data_sequence, targets):
        """
        Entraînement avec fenêtre glissante
        data_sequence: [batch, seq_len, channels, height, width] 
        targets: [batch, seq_len-4, channels, height, width] (car on prédit à partir de t=5)
        """
        batch_size, seq_len = data_sequence.shape[:2]
        
        model.reset_buffers()
        total_loss = 0
        valid_predictions = 0
        
        for t in range(seq_len):
            x_t = data_sequence[:, t]
            
            prediction = model(x_t, return_prediction=True)
            
            if prediction is not None and t-4 < targets.shape[1]:  # On a 4 frames de "warmup"
                target = targets[:, t-4]  # Target correspondant
                loss = torch.nn.functional.mse_loss(prediction, target)
                
                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                valid_predictions += 1
        
        avg_loss = total_loss / valid_predictions if valid_predictions > 0 else 0
        return avg_loss
    
    # Exemple d'entraînement
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Données d'exemple
    train_sequence = torch.randn(batch_size, 10, channels, height, width).to(device)
    train_targets = train_sequence[:, 4:]  # Les targets sont les frames futures
    
    # Un pas d'entraînement
    model.train()
    loss = train_step_windowed(model, optimizer, train_sequence, train_targets)
    print(f"Loss d'entraînement: {loss:.4f}")

if __name__ == "__main__":
    demo_temporal_rollout()
