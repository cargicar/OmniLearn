import torch
import torch.nn as nn
from torch.nn import functional as F

# Custom layers (replace with your implementation)
from StochasticDepth import StochasticDepth
from TalkingHeadAttention import TalkingHeadAttention
from LayerScale import LayerScale
from RandomDrop import RandomDrop

class PET(nn.Module):
  """Point-Edge Transformer"""
  def __init__(self,
              num_feat,
              num_jet,
              num_classes=2,
              num_keep = 7, #Number of features that wont be dropped
              feature_drop = 0.1,
              projection_dim = 128,
              local = True, K = 10,
              num_local = 2, 
              num_layers = 8, num_class_layers=2,
              num_gen_layers = 2,
              num_heads = 4,drop_probability = 0.0,
              simple = False, layer_scale = True,
              layer_scale_init = 1e-5,
              talking_head = False,
              mode = 'classifier',
              num_diffusion = 3,
              dropout=0.0,
              class_activation=None,
              ):
    super(PET, self).__init__()
    self.num_feat = num_feat
    self.num_jet = num_jet
    self.num_classes = num_classes
    self.num_keep = num_keep
    self.feature_drop = feature_drop
    self.drop_probability = drop_probability
    self.dropout = dropout
    self.projection_dim = projection_dim
    self.num_heads = num_heads
    self.num_layers = num_layers
    self.layer_scale = layer_scale
    self.layer_scale_init = layer_scale_init
    self.mode = mode
    self.num_diffusion = num_diffusion
    self.ema = 0.999
    self.class_activation = class_activation


    # Replace with your implementation of these modules
    self.stochastic_depth = StochasticDepth(drop_probability)
    self.talking_head_attention = TalkingHeadAttention(num_heads, projection_dim)
    self.layer_scale = LayerScale(layer_scale_init, dim=projection_dim)
    self.random_drop = RandomDrop(feature_drop, num_keep=num_keep)

    # Encoder layers
    self.encoder_layers = nn.ModuleList([
        self._make_encoding_layer() for _ in range(num_layers)
    ])

    # Classifier head layers
    self.classifier_layers = nn.ModuleList([
        self._make_classifier_layer() for _ in range(num_class_layers)
    ])

    # Generator layers
    self.generator_layers = nn.ModuleList([
        self._make_generator_layer() for _ in range(num_gen_layers)
    ])

    self.pred_tracker = torch.nn.modules.accuracy.Accuracy(dim=1)
    self.loss_tracker = torch.nn.modules.Loss(reduction='mean')
    self.mse_tracker = torch.nn.modules.MSELoss(reduction='mean')
    self.gen_tracker = torch.nn.modules.MSELoss(reduction='mean')
    self.pred_smear_tracker = torch.nn.modules.accuracy.Accuracy(dim=1)
    self.mse_smear_tracker = torch.nn.modules.MSELoss(reduction='mean')

  def _make_encoding_layer(self):
    return nn.Sequential(
        self.stochastic_depth(
            nn.Sequential(
                self.talking_head_attention(self.projection_dim, self.num_heads),
                self.layer)))
  
  import torch
import torch.nn as nn

class PET_body(nn.Module):
    def __init__(self, 
                 num_feat, 
                 projection_dim, 
                 num_layers, 
                 num_heads, 
                 drop_probability, 
                 layer_scale, 
                 layer_scale_init, 
                 talking_head, 
                 local, 
                 K, 
                 num_local, 
                 feature_drop, 
                 num_keep):
        super(PET_body, self).__init__()
        self.num_feat = num_feat
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.drop_probability = drop_probability
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init
        self.talking_head = talking_head
        self.local = local
        self.K = K
        self.num_local = num_local
        self.feature_drop = feature_drop
        self.num_keep = num_keep

        self.layers = nn.ModuleList([
            EncoderLayer(
                projection_dim=projection_dim, 
                num_heads=num_heads, 
                drop_probability=drop_probability, 
                layer_scale=layer_scale, 
                layer_scale_init=layer_scale_init, 
                talking_head=talking_head
            ) for _ in range(num_layers)
        ])

    def forward(self, input_features, input_points, input_mask, input_time):
        # Randomly drop features not present in other datasets
        encoded = RandomDrop(self.feature_drop, num_skip=self.num_keep)(input_features)
        encoded = get_encoding(encoded, self.projection_dim)

        # Time encoding
        time = FourierProjection(input_time, self.projection_dim)
        time = time[:, None, :].repeat(1, encoded.shape[1], 1) * input_mask
        time = nn.Linear(self.projection_dim, 2*self.projection_dim, bias=False)(time)
        scale, shift = torch.split(time, self.projection_dim, dim=-1)
        encoded = encoded * (1.0 + scale) + shift

        # Local attention (if enabled)
        if self.local:
            coord_shift = 999. * (1.0 - input_mask)  # Mask out padded positions
            points = input_points[:, :, :2]
            local_features = input_features
            for _ in range(self.num_local):
                local_features = get_neighbors(coord_shift + points, local_features, self.projection_dim, self.K)
                points = local_features
            encoded = encoded + local_features

        # Skip connection
        skip_connection = encoded

        # Transformer encoder layers
        for layer in self.layers:
            encoded = layer(encoded, input_mask)

        return encoded + skip_connection

class EncoderLayer(nn.Module):
    def __init__(self, projection_dim, num_heads, drop_probability, layer_scale, layer_scale_init, talking_head):
        super(EncoderLayer, self).__init__()
        self.self_attention = TalkingHeadAttention(projection_dim, num_heads, 0.0) if talking_head else \
                               nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads) 
        self.layer_norm1 = nn.LayerNorm(projection_dim)
        self.dropout1 = nn.Dropout(drop_probability)
        self.linear1 = nn.Linear(projection_dim, 2*projection_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(drop_probability)
        self.linear2 = nn.Linear(2*projection_dim, projection_dim)
        self.layer_norm2 = nn.LayerNorm(projection_dim)
        self.layer_scale = LayerScale(layer_scale_init, dim=projection_dim) if layer_scale else nn.Identity()

    def forward(self, x, mask):
        # Self-attention
        residual = x
        x, _ = self.self_attention(x, x, x, key_padding_mask=~mask.bool())
        x = self.dropout1(x)
        x = residual + x
        x = self.layer_norm1(x)

        # Feed-forward network
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.layer_scale(x, mask) 
        x = residual + x
        x = self.layer_norm2(x) 

        return x

# ... (Other helper functions: get_encoding, FourierProjection, get_neighbors, etc.) ...
######################################################################
# Example usage:
# Assuming you have defined RandomDrop, TalkingHeadAttention, 
# LayerScale, get_encoding, FourierProjection, and get_neighbors 
# as PyTorch modules/functions

# num_feat = 128  # Example
# projection_dim = 256
# num_layers = 6
# num_heads = 4
# drop_probability = 0.1
# layer_scale = True
# layer_scale_init = 1e-5
# talking_head = False 
# local = True
# K = 10
# num_local = 2
# feature_drop = 0.1
# num_keep = 7

# pet_body = PET_body(
#     num_feat=num_feat, 
#     projection_dim=projection_dim, 
#     num_layers=num_layers, 
#     num_heads=num_heads, 
#     drop_probability=drop_probability, 
#     layer_scale=layer_scale, 
#     layer_scale_init=layer_scale_init, 
#     talking_head=talking_head, 
#     local=local, 
#     K=K, 
#     num_local=num_local, 
#     feature_drop=feature_drop, 
#     num_keep=num_keep
# )

# # Sample input tensors (replace with your actual data)
# input_features = torch.randn(batch_size, seq_len, num_feat)
# input_points = torch.randn(batch_size, seq_len, 2)
# input_mask = torch.ones(batch_size, seq_len, 1) 
# input_time = torch.randn(batch_size, 1) 

# output = pet_body(input_features, input_points, input_mask, input_time)
#################################################################################