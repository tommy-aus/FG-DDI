import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Set2Set,
                                )

from layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    )
import time


class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features  # 55
        self.hidd_dim = hidd_dim  # 128
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        
        # Initialize normalization for initial features
        self.initial_norm = LayerNorm(self.in_features)
        
        # Initialize blocks with correct dimensions
        self.blocks = nn.ModuleList()
        current_dim = in_features  # Start with 55 features
        
        for i, (n_heads, head_out_feats) in enumerate(zip(blocks_params, heads_out_feat_params)):
            block = MVN_DDI_Block(n_heads, current_dim, head_out_feats, self.hidd_dim)
            self.blocks.append(block)
            current_dim = head_out_feats * 2  # Update for next block's input
            
        self.net_norms = nn.ModuleList([
            LayerNorm(head_out_feats * 2)  # *2 because of concatenation
            for head_out_feats in heads_out_feat_params
        ])
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        
    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples
        
        # Normalize initial features
        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        
        repr_h = []
        repr_t = []
        
        for i, (block, norm) in enumerate(zip(self.blocks, self.net_norms)):
            h_data, t_data, r_h, r_t = block(h_data, t_data, b_graph)
            
            repr_h.append(r_h)
            repr_t.append(r_t)
            
            # Apply normalization
            h_data.x = F.elu(norm(h_data.x, h_data.batch))
            t_data.x = F.elu(norm(t_data.x, t_data.batch))
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        
        attentions = self.co_attention(repr_h, repr_t)
        scores = self.KGE(repr_h, repr_t, rels, attentions)
        
        return scores


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        
        # Initial feature transformation
        # Input: in_features (starts at 55, then 128 for subsequent blocks)
        # Output per head: head_out_feats // n_heads
        # Total output after concatenation: head_out_feats
        self.feature_conv = GATConv(in_features, head_out_feats // n_heads, n_heads)
        
        # Both attention mechanisms take head_out_feats as input
        self.intraAtt = IntraGraphAttention(head_out_feats)
        self.interAtt = InterGraphAttention(head_out_feats)
        
        # Takes concatenated features as input (head_out_feats * 2)
        self.readout = SAGPooling(head_out_feats * 2, min_score=-1)
    
    def forward(self, h_data, t_data, b_graph):
        # Transform features - input dim varies, output is head_out_feats
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)
        
        # Both attention mechanisms work with head_out_feats dimensions
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        
        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)
        
        # Concatenate to double the features
        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep
        
        # Process the concatenated features with safe global pooling
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(
            h_data.x, h_data.edge_index, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(
            t_data.x, t_data.edge_index, batch=t_data.batch)
        
        # Safe global pooling with empty batch handling
        h_global_graph_emb = self.safe_global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = self.safe_global_add_pool(t_att_x, t_att_batch)
        
        return h_data, t_data, h_global_graph_emb, t_global_graph_emb

    def safe_global_add_pool(self, x, batch):
        """
        Safe version of global_add_pool that handles empty batches
        """
        if x.numel() == 0 or batch.numel() == 0:
            # Return zero tensor with appropriate shape
            return torch.zeros(1, x.size(-1), device=x.device, dtype=x.dtype)
        
        # Check if batch is valid
        if batch.max() < 0:
            return torch.zeros(1, x.size(-1), device=x.device, dtype=x.dtype)
            
        return global_add_pool(x, batch)