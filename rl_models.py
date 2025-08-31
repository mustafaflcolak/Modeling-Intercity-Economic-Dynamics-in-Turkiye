#!/usr/bin/env python3
"""
RL model definitions for the Turkey Economy simulation.
This file contains the neural network models used for RL policy decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PolicyNetwork(nn.Module):
    """
    Neural network that takes graph-structured observations and outputs policy probabilities.
    Uses the architecture that matches the saved checkpoint file.
    """
    
    def __init__(self, num_cities: int = 81, feature_dim: int = 13, global_dim: int = 3):
        super().__init__()
        
        # Store dimensions
        self.num_cities = num_cities
        self.feature_dim = feature_dim
        self.global_dim = global_dim
        
        # City feature encoder (per-city MLP)
        self.city_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Policy head (MultiDiscrete: [6, 3, 81, 4])
        self.policy_family_head = nn.Sequential(
            nn.Linear(64 * num_cities + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 6 policy families (as per checkpoint)
        )
        
        self.scope_head = nn.Sequential(
            nn.Linear(64 * num_cities + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 scopes
        )
        
        self.target_head = nn.Sequential(
            nn.Linear(64 * num_cities + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_cities),  # 81 cities
        )
        
        self.strength_head = nn.Sequential(
            nn.Linear(64 * num_cities + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 strength levels (as per checkpoint)
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(64 * num_cities + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract features
        node_features = obs.get('node_features', torch.zeros(1, self.num_cities, self.feature_dim))
        global_features = obs.get('global_features', torch.zeros(1, self.global_dim))
        
        batch_size = node_features.shape[0]
        num_cities = node_features.shape[1]
        
        # Encode city features
        city_encoded = self.city_encoder(node_features)  # [batch, num_cities, 64]
        city_flat = city_encoded.view(batch_size, -1)  # [batch, num_cities * 64]
        
        # Encode global features
        global_encoded = self.global_encoder(global_features)  # [batch, 32]
        
        # Combine features
        combined = torch.cat([city_flat, global_encoded], dim=1)  # [batch, num_cities * 64 + 32]
        
        # Policy outputs
        policy_family_logits = self.policy_family_head(combined)
        scope_logits = self.scope_head(combined)
        target_logits = self.target_head(combined)
        strength_logits = self.strength_head(combined)
        
        return policy_family_logits, scope_logits, target_logits, strength_logits
    
    def get_action(self, obs: Dict[str, torch.Tensor], deterministic: bool = True) -> Tuple[int, int, int, int]:
        """
        Get action from the policy network.
        
        Args:
            obs: Observation dictionary
            deterministic: If True, return argmax action. If False, sample from distribution.
        
        Returns:
            Tuple of (policy_family, scope, target_city, strength)
        """
        with torch.no_grad():
            policy_family_logits, scope_logits, target_logits, strength_logits = self.forward(obs)
            
            if deterministic:
                # Return argmax actions
                policy_family = torch.argmax(policy_family_logits, dim=1).item()
                scope = torch.argmax(scope_logits, dim=1).item()
                target_city = torch.argmax(target_logits, dim=1).item()
                strength = torch.argmax(strength_logits, dim=1).item()
            else:
                # Sample from categorical distributions
                policy_family_dist = torch.distributions.Categorical(logits=policy_family_logits)
                scope_dist = torch.distributions.Categorical(logits=scope_logits)
                target_dist = torch.distributions.Categorical(logits=target_logits)
                strength_dist = torch.distributions.Categorical(logits=strength_logits)
                
                policy_family = policy_family_dist.sample().item()
                scope = scope_dist.sample().item()
                target_city = target_dist.sample().item()
                strength = strength_dist.sample().item()
            
            return policy_family, scope, target_city, strength


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values in RL training.
    """
    
    def __init__(self, num_cities: int = 81, feature_dim: int = 13, global_dim: int = 3):
        super(ValueNetwork, self).__init__()
        
        # Store dimensions as attributes
        self.num_cities = num_cities
        self.feature_dim = feature_dim
        self.global_dim = global_dim
        
        # Same architecture as policy network but with single output
        total_input_size = (num_cities * feature_dim) + global_dim
        hidden_size = 256
        
        self.fc1 = nn.Linear(total_input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass to estimate state value."""
        # Extract features
        node_features = obs.get('node_features', torch.zeros(1, self.num_cities, self.feature_dim))
        global_features = obs.get('global_features', torch.zeros(1, self.global_dim))
        
        # Flatten and concatenate
        batch_size = node_features.size(0)
        node_features_flat = node_features.view(batch_size, -1)
        combined_features = torch.cat([node_features_flat, global_features], dim=1)
        
        # Forward pass
        x = F.relu(self.layer_norm1(self.fc1(combined_features)))
        x = self.dropout(x)
        
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.layer_norm3(self.fc3(x)))
        x = self.dropout(x)
        
        # Value output
        value = self.value_head(x)
        return value


def create_rl_models(num_cities: int = 81, feature_dim: int = 13, global_dim: int = 3):
    """
    Factory function to create RL models.
    
    Args:
        num_cities: Number of cities in the simulation
        feature_dim: Number of features per city
        global_dim: Number of global features
    
    Returns:
        Tuple of (policy_network, value_network)
    """
    policy_net = PolicyNetwork(num_cities, feature_dim, global_dim)
    value_net = ValueNetwork(num_cities, feature_dim, global_dim)
    
    return policy_net, value_net


def load_rl_model(checkpoint_path: str, device: str = "auto") -> PolicyNetwork:
    """
    Load a trained RL model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on ("auto", "cuda", "cpu", "mps")
    
    Returns:
        Loaded PolicyNetwork
    """
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    
    # Load checkpoint with compatibility handling for PyTorch 2.6+
    try:
        # First try with weights_only=True (PyTorch 2.6+ default)
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=True)
        print("✅ RL model loaded successfully with weights_only=True")
    except Exception as e:
        # If that fails, try with weights_only=False for older checkpoint compatibility
        try:
            print("⚠️  weights_only=True failed, trying weights_only=False for compatibility...")
            checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
            print("✅ RL model loaded successfully with weights_only=False (older checkpoint format)")
        except Exception as e2:
            raise ValueError(f"Failed to load checkpoint with both methods: {e2}")
    
    # Create model
    model = PolicyNetwork()
    model.to(device_obj)
    
    # Load weights
    if 'policy_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['policy_state_dict'])
        print(f"✅ RL model loaded successfully from checkpoint")
        print(f"   Episode: {checkpoint.get('episode', 'Unknown')}")
        print(f"   Best reward: {checkpoint.get('best_reward', 'Unknown')}")
    else:
        # Try loading directly if it's just the model
        try:
            model.load_state_dict(checkpoint)
            print("✅ RL model loaded successfully (direct state dict)")
        except Exception as e:
            raise ValueError(f"Invalid checkpoint format: {e}")
    
    model.eval()  # Set to evaluation mode
    return model 