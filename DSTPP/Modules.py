import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, temperature, attn_dropout=0.2, use_hawkes_prior=True, loc_dim=2, 
                 fusion_mode='log_add', adaptive_fusion=True, separate_temporal_spatial=True):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.use_hawkes_prior = use_hawkes_prior
        self.loc_dim = loc_dim
        self.fusion_mode = fusion_mode  # 'linear', 'log_add', 'gated', 'multiplicative'
        self.adaptive_fusion = adaptive_fusion
        self.separate_temporal_spatial = separate_temporal_spatial
        
        if use_hawkes_prior:
            self.time_decay = nn.Parameter(torch.ones(1) * 0.1)  
            self.space_sigma = nn.Parameter(torch.ones(1) * 0.5)  
            
            if separate_temporal_spatial:
                self.temporal_strength = nn.Parameter(torch.ones(1) * 0.1)  
                self.spatial_strength = nn.Parameter(torch.ones(1) * 0.1)   
            else:
                self.fusion_alpha = nn.Parameter(torch.ones(1) * 0.1)  
            
            if adaptive_fusion:
                self.gate_network = nn.Sequential(
                    nn.Linear(2, 16), 
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    nn.Sigmoid()
                )
            
            self.time_embed = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

            if loc_dim >= 2:
                self.space_embed = nn.Sequential(
                    nn.Linear(1, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1)
                )

    def compute_hawkes_prior(self, event_time, event_loc, device):

        batch_size, seq_len = event_time.shape
        

        delta_t = event_time.unsqueeze(1) - event_time.unsqueeze(2)  # [batch, seq, seq]
        delta_t = F.relu(delta_t)  
        

        if event_loc is not None and event_loc.dim() == 3:
            actual_loc_dim = event_loc.shape[-1]
            loc_diff = event_loc.unsqueeze(1) - event_loc.unsqueeze(2)  # [batch, seq, seq, loc_dim]
            
            if actual_loc_dim == 1:
                delta_s = torch.abs(loc_diff.squeeze(-1))
            else:
                delta_s = torch.norm(loc_diff, dim=-1)
        else:
            delta_s = torch.zeros_like(delta_t)
        
        time_decay = F.softplus(self.time_decay) + 1e-4
        time_base = torch.exp(-time_decay * delta_t)
        

        if seq_len > 100:

            time_enhanced = torch.sigmoid(-delta_t * 0.1) * 0.1 + 1.0  
            temporal_prior = time_base * time_enhanced
            del time_base, time_enhanced
        else:
            delta_t_embedded = self.time_embed(delta_t.unsqueeze(-1)).squeeze(-1)
            time_enhanced = torch.sigmoid(delta_t_embedded) * 0.1 + 1.0  # 降低增强因子（从0.3到0.1）
            temporal_prior = time_base * time_enhanced
            del delta_t_embedded, time_base, time_enhanced

        temporal_prior = F.softmax(temporal_prior / 0.1, dim=-1)  

        if event_loc is not None and delta_s.max() > 1e-6:
            space_sigma = F.softplus(self.space_sigma) + 0.1
            space_base = torch.exp(-(delta_s ** 2) / (2 * space_sigma ** 2))
            if seq_len > 100:
                space_enhanced = torch.sigmoid(-delta_s * 0.1) * 0.1 + 1.0  # 轻量级增强
                spatial_prior = space_base * space_enhanced
                del space_base, space_enhanced
            elif self.loc_dim >= 2 and hasattr(self, 'space_embed'):
                delta_s_embedded = self.space_embed(delta_s.unsqueeze(-1)).squeeze(-1)
                space_enhanced = torch.sigmoid(delta_s_embedded) * 0.1 + 1.0  # 降低增强因子
                spatial_prior = space_base * space_enhanced
                del delta_s_embedded, space_base, space_enhanced
            else:
                spatial_prior = space_base
                del space_base
            spatial_prior = F.softmax(spatial_prior / 0.1, dim=-1) 
            del delta_s
        else:
            spatial_prior = torch.ones_like(temporal_prior) / seq_len  
        del delta_t
        if 'loc_diff' in locals():
            del loc_diff
        
        return temporal_prior, spatial_prior

    def fuse_attention_with_prior(self, attn_logits, temporal_prior, spatial_prior):
        batch_size, num_heads, seq_len, _ = attn_logits.shape
        
        if self.separate_temporal_spatial:
            temporal_strength = torch.sigmoid(self.temporal_strength)
            spatial_strength = torch.sigmoid(self.spatial_strength)
            temporal_prior = temporal_prior.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [batch, heads, seq, seq]
            spatial_prior = spatial_prior.unsqueeze(1).expand(-1, num_heads, -1, -1)    # [batch, heads, seq, seq]
        else:
            alpha = torch.sigmoid(self.fusion_alpha)
            combined_prior = alpha * temporal_prior + (1 - alpha) * spatial_prior
            combined_prior = combined_prior.unsqueeze(1).expand(-1, num_heads, -1, -1)  # [batch, heads, seq, seq]
            temporal_prior = combined_prior
            spatial_prior = combined_prior
            temporal_strength = torch.sigmoid(self.fusion_alpha)
            spatial_strength = torch.sigmoid(self.fusion_alpha)
        
        if self.fusion_mode == 'log_add':
            log_temporal_prior = torch.log(temporal_prior + 1e-8)
            log_spatial_prior = torch.log(spatial_prior + 1e-8)
            fused_logits = attn_logits + temporal_strength * log_temporal_prior + spatial_strength * log_spatial_prior
            fused_attn = F.softmax(fused_logits, dim=-1)
            
        elif self.fusion_mode == 'multiplicative':
            attn_probs = F.softmax(attn_logits, dim=-1)
            fused_attn = (attn_probs ** (1 - temporal_strength - spatial_strength)) * \
                        (temporal_prior ** temporal_strength) * \
                        (spatial_prior ** spatial_strength)
            fused_attn = fused_attn / (fused_attn.sum(dim=-1, keepdim=True) + 1e-8)
            
        elif self.fusion_mode == 'gated':
            attn_probs = F.softmax(attn_logits, dim=-1)
            
            attn_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1, keepdim=True)  # [batch, heads, seq, 1]
            attn_entropy_flat = attn_entropy.view(batch_size * num_heads, seq_len)
            attn_entropy_norm = (attn_entropy_flat - attn_entropy_flat.min(dim=-1, keepdim=True)[0]) / \
                               (attn_entropy_flat.max(dim=-1, keepdim=True)[0] - attn_entropy_flat.min(dim=-1, keepdim=True)[0] + 1e-8)
            attn_entropy_norm = attn_entropy_norm.view(batch_size, num_heads, seq_len, 1)
            prior_std = torch.std(temporal_prior, dim=-1, keepdim=True)  # [batch, 1, seq, 1]
            prior_confidence = 1.0 - torch.clamp(prior_std, 0, 1)
            prior_confidence = prior_confidence.expand(-1, num_heads, -1, -1)  # [batch, heads, seq, 1]

            gate_input = torch.cat([attn_entropy_norm, prior_confidence], dim=-1)  # [batch, heads, seq, 2]
            gate_input_flat = gate_input.view(batch_size * num_heads * seq_len, 2)
            gate_weight_flat = self.gate_network(gate_input_flat)  # [batch*heads*seq, 1]
            gate_weight = gate_weight_flat.view(batch_size, num_heads, seq_len, 1)

            combined_prior = temporal_strength * temporal_prior + spatial_strength * spatial_prior
            fused_attn = gate_weight * combined_prior + (1 - gate_weight) * attn_probs
            fused_attn = fused_attn / (fused_attn.sum(dim=-1, keepdim=True) + 1e-8)
            
        elif self.fusion_mode == 'linear':
            attn_probs = F.softmax(attn_logits, dim=-1)
            fused_attn = (1 - temporal_strength - spatial_strength) * attn_probs + \
                        temporal_strength * temporal_prior + \
                        spatial_strength * spatial_prior
            fused_attn = fused_attn / (fused_attn.sum(dim=-1, keepdim=True) + 1e-8)
            
        else:
            attn_probs = F.softmax(attn_logits, dim=-1)
            log_attn = torch.log(attn_probs + 1e-8)
            log_temporal_prior = torch.log(temporal_prior + 1e-8)
            log_spatial_prior = torch.log(spatial_prior + 1e-8)
            fused_log = log_attn + temporal_strength * log_temporal_prior + spatial_strength * log_spatial_prior
            fused_attn = F.softmax(fused_log, dim=-1)
        
        return fused_attn

    def forward(self, q, k, v, mask=None, event_time=None, event_loc=None, diffusion_step=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        batch_size, num_heads, seq_len, _ = attn.shape
        
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        if self.use_hawkes_prior and event_time is not None:
            temporal_prior, spatial_prior = self.compute_hawkes_prior(
                event_time, event_loc, attn.device
            )
            attn_weights = self.fuse_attention_with_prior(attn, temporal_prior, spatial_prior)
        else:
            attn_weights = F.softmax(attn, dim=-1)
        
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
