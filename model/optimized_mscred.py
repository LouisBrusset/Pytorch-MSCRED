import torch
import torch.nn as nn
from collections import deque
import numpy as np
from model.convolution_lstm import ConvLSTMCell

class OptimizedMSCRED(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder, sequence_length=5):
        super(OptimizedMSCRED, self).__init__()
        self.sequence_length = sequence_length
        
        # CNN Encoder (partagé, calculé une fois par frame)
        self.cnn_encoder = CnnEncoder(in_channels_encoder)
        
        # ConvLSTM pour chaque niveau de features (optimisé)
        self.conv1_lstm = OptimizedConvLSTM(32, 32, 3)
        self.conv2_lstm = OptimizedConvLSTM(64, 64, 3)
        self.conv3_lstm = OptimizedConvLSTM(128, 128, 3)
        self.conv4_lstm = OptimizedConvLSTM(256, 256, 3)
        
        # CNN Decoder
        self.cnn_decoder = CnnDecoder(in_channels_decoder)
        
        # Buffers circulaires pour stocker les features encodées
        self.feature_buffers = {
            'conv1': deque(maxlen=sequence_length),
            'conv2': deque(maxlen=sequence_length), 
            'conv3': deque(maxlen=sequence_length),
            'conv4': deque(maxlen=sequence_length)
        }
        
        self.initialized = False

    def forward(self, x_new, return_prediction=True):
        """
        x_new: nouvelle frame [batch, channels, height, width]
        return_prediction: si True, retourne une prédiction (nécessite buffer plein)
        """
        batch_size = x_new.size(0)
        
        # 1. ENCODE la nouvelle frame UNE SEULE FOIS
        conv1_new, conv2_new, conv3_new, conv4_new = self.cnn_encoder(x_new)
        
        # 2. AJOUTER aux buffers circulaires
        self.feature_buffers['conv1'].append(conv1_new.detach())
        self.feature_buffers['conv2'].append(conv2_new.detach()) 
        self.feature_buffers['conv3'].append(conv3_new.detach())
        self.feature_buffers['conv4'].append(conv4_new.detach())
        
        # 3. VÉRIFIER si on peut faire une prédiction
        if len(self.feature_buffers['conv1']) < self.sequence_length:
            if return_prediction:
                return None  # Pas assez de données pour prédire
            else:
                return conv1_new, conv2_new, conv3_new, conv4_new
        
        # 4. CRÉER les séquences temporelles pour le LSTM
        conv1_sequence = torch.stack(list(self.feature_buffers['conv1']), dim=1)  # [batch, seq, channels, h, w]
        conv2_sequence = torch.stack(list(self.feature_buffers['conv2']), dim=1)
        conv3_sequence = torch.stack(list(self.feature_buffers['conv3']), dim=1) 
        conv4_sequence = torch.stack(list(self.feature_buffers['conv4']), dim=1)
        
        # 5. APPLIQUER ConvLSTM sur les séquences
        conv1_lstm_out = self.conv1_lstm(conv1_sequence)
        conv2_lstm_out = self.conv2_lstm(conv2_sequence)
        conv3_lstm_out = self.conv3_lstm(conv3_sequence)
        conv4_lstm_out = self.conv4_lstm(conv4_sequence)
        
        if return_prediction:
            # 6. DECODER pour génération
            gen_x = self.cnn_decoder(conv1_lstm_out, conv2_lstm_out, 
                                   conv3_lstm_out, conv4_lstm_out)
            return gen_x
        else:
            return conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out

    def reset_buffers(self):
        """Reset les buffers pour une nouvelle séquence"""
        for buffer in self.feature_buffers.values():
            buffer.clear()
        self.initialized = False

class OptimizedConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(OptimizedConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # ConvLSTM Cell
        self.lstm_cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        
    def forward(self, sequence):
        """
        sequence: [batch, seq_len, channels, height, width]
        """
        batch_size, seq_len, channels, height, width = sequence.size()
        
        # Initialiser les états cachés
        h, c = self.lstm_cell.init_hidden(batch_size, self.hidden_channels, (height, width))
        
        outputs = []
        
        # Rollout temporel
        for t in range(seq_len):
            x_t = sequence[:, t]  # [batch, channels, height, width]
            h, c = self.lstm_cell(x_t, h, c)
            outputs.append(h)
            
        # Retourner seulement la dernière sortie (ou toutes selon le besoin)
        return torch.stack(outputs, dim=1)[:, -1]  # [batch, hidden_channels, height, width]

# Classes existantes à adapter...
class CnnEncoder(nn.Module):
    def __init__(self, in_channels_encoder):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_encoder, 32, 3, (1, 1), 1),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, (2, 2), 1),
            nn.SELU()
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, (2, 2), 1),
            nn.SELU()
        )   
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, (2, 2), 0),
            nn.SELU()
        )
    
    def forward(self, X):
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out, conv4_out

class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 0),
            nn.SELU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2, 1, 1),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 2, 1, 1),
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, 1, 1, 0),
            nn.SELU()
        )
    
    def forward(self, conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        # Ajouter dimension batch si nécessaire
        conv1_lstm_out = conv1_lstm_out.unsqueeze(0) if conv1_lstm_out.dim() == 3 else conv1_lstm_out
        conv2_lstm_out = conv2_lstm_out.unsqueeze(0) if conv2_lstm_out.dim() == 3 else conv2_lstm_out
        conv3_lstm_out = conv3_lstm_out.unsqueeze(0) if conv3_lstm_out.dim() == 3 else conv3_lstm_out
        conv4_lstm_out = conv4_lstm_out.unsqueeze(0) if conv4_lstm_out.dim() == 3 else conv4_lstm_out
        
        deconv4 = self.deconv4(conv4_lstm_out)
        deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim=1)
        deconv3 = self.deconv3(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim=1)
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim=1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1
