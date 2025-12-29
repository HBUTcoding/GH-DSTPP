import torch
import torch.nn as nn  


class TemporalBackgroundEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.time_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
       
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        

        self.temporal_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        

        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, event_time, non_pad_mask):

        time_emb = self.time_embedding(event_time.unsqueeze(-1))
        time_emb = time_emb * non_pad_mask
        
        lengths = non_pad_mask.squeeze(-1).sum(dim=1).long().cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            time_emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.bilstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_output = lstm_output * non_pad_mask
        

        batch_size, seq_len = event_time.shape
        attn_mask = ~non_pad_mask.squeeze(-1).bool()  # [batch, seq]
        
        attn_output, attn_weights = self.temporal_self_attention(
            lstm_output, lstm_output, lstm_output,
            key_padding_mask=attn_mask
        )
        
        output = self.layer_norm(lstm_output + attn_output)
        output = output * non_pad_mask
        
        background_feat = self.projection(output)
        return background_feat