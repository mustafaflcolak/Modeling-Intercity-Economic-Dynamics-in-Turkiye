import os
import time
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from datetime import datetime
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.distributions import Categorical

from rl_env import TurkeyPolicyEnv
from rl_models import PolicyNetwork, ValueNetwork, create_rl_models


def check_gpu_availability():
    """Check GPU availability and provide detailed information."""
    print("=" * 60)
    print("GPU/CUDA AVAILABILITY CHECK")
    print("=" * 60)
    
    # Check PyTorch CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Show GPU information
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check current device
        current_device = torch.cuda.current_device()
        print(f"Current GPU device: {current_device}")
        
        # Test GPU computation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("✅ GPU computation test: PASSED")
            print(f"   Test tensor shape: {test_tensor.shape}")
            print(f"   Test tensor device: {test_tensor.device}")
            
            # Clean up test tensor
            del test_tensor, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ GPU computation test: FAILED - {e}")
            return False
        
        print("✅ CUDA is working properly!")
        return True
        
    else:
        print("❌ CUDA is not available")
        
        # Check alternative backends
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) is available")
            return "mps"
        else:
            print("❌ No GPU acceleration available - using CPU")
            print("   Training will be slower on CPU")
            return False
    
    print("=" * 60)


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()


class ProgressLogger:
    """Logs training progress to a text file for analysis."""
    
    def __init__(self, log_file: str = "training_progress.txt"):
        self.log_file = log_file
        self.start_time = datetime.now()
        
        # Create header
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TURKEY POLICY RL TRAINING PROGRESS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Episode':>8} | {'Reward':>10} | {'Avg_Reward':>12} | {'Best_Reward':>12} | {'Steps':>6} | {'Loss':>10} | {'Policy_Loss':>12} | {'Value_Loss':>11} | {'Entropy_Loss':>12} | {'Time':>10}\n")
            f.write("-" * 80 + "\n")
    
    def log_episode(self, episode: int, reward: float, avg_reward: float, best_reward: float, 
                    steps: int, loss_info: Dict[str, float], episode_time: float):
        """Log one episode's training statistics."""
        total_loss = loss_info.get('total_loss', 0.0)
        policy_loss = loss_info.get('policy_loss', 0.0)
        value_loss = loss_info.get('value_loss', 0.0)
        entropy_loss = loss_info.get('entropy_loss', 0.0)
        
        with open(self.log_file, 'a') as f:
            f.write(f"{episode:8d} | {reward:10.3f} | {avg_reward:12.3f} | {best_reward:12.3f} | {steps:6d} | {total_loss:10.6f} | {policy_loss:12.6f} | {value_loss:11.6f} | {entropy_loss:12.6f} | {episode_time:10.2f}s\n")
    
    def log_checkpoint(self, episode: int, checkpoint_path: str, is_best: bool):
        """Log checkpoint creation."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CHECKPOINT SAVED - Episode {episode}\n")
            f.write(f"Path: {checkpoint_path}\n")
            f.write(f"Best Model: {'YES' if is_best else 'NO'}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"{'Episode':>8} | {'Reward':>10} | {'Avg_Reward':>12} | {'Best_Reward':>12} | {'Steps':>6} | {'Loss':>10} | {'Policy_Loss':>12} | {'Value_Loss':>11} | {'Entropy_Loss':>12} | {'Time':>10}\n")
            f.write("-" * 80 + "\n")
    
    def log_summary(self, total_episodes: int, total_steps: int, best_reward: float, 
                   avg_reward: float, training_time: float):
        """Log training completion summary."""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write("TRAINING COMPLETED\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total Episodes: {total_episodes}\n")
            f.write(f"Total Steps: {total_steps}\n")
            f.write(f"Best Reward: {best_reward:.3f}\n")
            f.write(f"Final Average Reward: {avg_reward:.3f}\n")
            f.write(f"Total Training Time: {training_time/3600:.2f} hours\n")
            f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")


class PPOTrainer:
    """
    ENHANCED PPO trainer for the Turkey policy environment.
    Handles training, checkpointing, and resuming from interruptions with improved strategies.
    """
    
    def __init__(
        self,
        env: TurkeyPolicyEnv,
        learning_rate: float = 3e-4,
        gamma: float = 0.985,
        gae_lambda: float = 0.97,
        clip_epsilon: float = 0.15,
        value_loss_coef: float = 0.6,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.3,
        ppo_epochs: int = 3,
        batch_size: int = 48,
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
    ):
        self.env = env
        self.device = self._get_device(device)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Progress logging
        self.logger = ProgressLogger()
        
        # Initialize networks
        self.policy_net, self.value_net = create_rl_models(
            num_cities=81,  # Fixed number of cities
            feature_dim=13,  # Fixed feature dimension
            global_dim=3     # Fixed global dimension
        )
        # Move networks to device
        self.policy_net = self.policy_net.to(self.device)
        self.value_net = self.value_net.to(self.device)
        
        # ENHANCED: Use AdamW optimizer with weight decay for better regularization
        self.optimizer = optim.AdamW(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,  # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # ENHANCED: Learning rate scheduler for better convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.8,
            patience=200,  # Reduce LR if no improvement for 200 episodes
            min_lr=1e-6
        )
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        
        # Experience buffers
        self.episode_rewards = deque(maxlen=200)  # INCREASED: Better statistics
        self.episode_lengths = deque(maxlen=200)
        
        # ENHANCED: Training statistics for monitoring
        self.training_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        
        # ENHANCED: Early stopping and performance tracking
        self.episodes_without_improvement = 0
        self.early_stopping_patience = 500  # Stop if no improvement for 500 episodes
        
        # Load checkpoint if exists
        self.load_checkpoint()
    
    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def save_checkpoint(self, episode: int, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size,
            }
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode:06d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        # Keep only last 5 checkpoints to save disk space
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_episode_*.pt"))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Log checkpoint to progress file
        self.logger.log_checkpoint(episode, str(checkpoint_path), is_best)
    
    def load_checkpoint(self) -> None:
        """Load latest checkpoint if exists."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_episode_*.pt"))
        if not checkpoints:
            print("No checkpoint found. Starting fresh training.")
            return
        
        latest_checkpoint = checkpoints[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        try:
            # Use weights_only=False for compatibility with older PyTorch versions
            checkpoint = torch.load(latest_checkpoint, map_location=self.device, weights_only=False)
            
            self.episode_count = checkpoint['episode']
            self.total_steps = checkpoint['total_steps']
            self.best_reward = checkpoint['best_reward']
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=200)
            self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=200)
            
            print(f"Resumed from episode {self.episode_count}, total steps: {self.total_steps}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Checkpoint file appears to be corrupted. Starting fresh training.")
            print("Removing corrupted checkpoint file...")
            
            try:
                latest_checkpoint.unlink()  # Delete corrupted file
                print("Corrupted checkpoint removed.")
            except Exception as del_error:
                print(f"Could not remove corrupted checkpoint: {del_error}")
            
            # Reset to fresh training
            self.episode_count = 0
            self.total_steps = 0
            self.best_reward = float('-inf')
            self.episode_rewards = deque(maxlen=200)
            self.episode_lengths = deque(maxlen=200)
    
    def collect_episode(self) -> Tuple[List[Dict], float, int]:
        """Collect one episode of experience."""
        obs = self.env.reset()
        episode_data = []
        total_reward = 0.0
        step_count = 0
        
        while True:
            # Convert observation to tensor
            obs_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0).to(self.device) 
                for k, v in obs.items()
            }
            
            # Get action from policy
            with torch.no_grad():
                policy_family_logits, scope_logits, target_logits, strength_logits = self.policy_net(obs_tensor)
                
                # Get value from value network
                value = self.value_net(obs_tensor)
                
                # Sample actions
                policy_family_dist = Categorical(logits=policy_family_logits)
                scope_dist = Categorical(logits=scope_logits)
                target_dist = Categorical(logits=target_logits)
                strength_dist = Categorical(logits=strength_logits)
                
                policy_family_action = policy_family_dist.sample()
                scope_action = scope_dist.sample()
                target_action = target_dist.sample()
                strength_action = strength_dist.sample()
                
                # Get log probabilities
                policy_family_log_prob = policy_family_dist.log_prob(policy_family_action)
                scope_log_prob = scope_dist.log_prob(scope_action)
                target_log_prob = target_dist.log_prob(target_action)
                strength_log_prob = strength_dist.log_prob(strength_action)
                
                # Combine into single action
                action = (
                    policy_family_action.item(),
                    scope_action.item(),
                    target_action.item(),
                    strength_action.item()
                )
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Store experience
            episode_data.append({
                'observation': obs,  # Changed from 'obs' to 'observation' for consistency
                'action': action,
                'reward': reward,
                'log_probs': (
                    policy_family_log_prob.item(),
                    scope_log_prob.item(),
                    target_log_prob.item(),
                    strength_log_prob.item()
                ),
                'value': value.item(),
                'done': done,
            })
            
            total_reward += reward
            step_count += 1
            obs = next_obs
            
            if done:
                break
        
        # Compute advantages for the episode
        advantages = self.compute_advantages(episode_data)
        
        # Add advantages to episode data
        for i, step in enumerate(episode_data):
            step['advantage'] = advantages[i]
        
        return episode_data, total_reward, step_count
    
    def compute_advantages(self, episode_data: List[Dict]) -> List[float]:
        """Compute GAE advantages for the episode."""
        if not episode_data:
            return []
            
        rewards = [step['reward'] for step in episode_data]
        values = [step['value'] for step in episode_data]
        
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update_policy(self, episode_data: List[Dict]) -> Dict[str, float]:
        """ENHANCED PPO update with better loss computation and training stability."""
        if len(episode_data) < self.batch_size:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
        
        # Sample batch from episode data
        batch_indices = np.random.choice(len(episode_data), self.batch_size, replace=False)
        batch_data = [episode_data[i] for i in batch_indices]
        
        # Prepare batch data
        obs_batch = []
        actions_batch = []
        old_log_probs_batch = []
        advantages_batch = []
        returns_batch = []
        
        for step in batch_data:
            obs_batch.append(step['observation'])
            actions_batch.append(step['action'])
            old_log_probs_batch.append(step['log_probs'])
            advantages_batch.append(step['advantage'])
            returns_batch.append(step['advantage'] + step['value'])
        
        # Convert to tensors
        obs_tensor = {
            'node_features': torch.FloatTensor(np.stack([o['node_features'] for o in obs_batch])).to(self.device),
            'global_features': torch.FloatTensor(np.stack([o['global_features'] for o in obs_batch])).to(self.device),
        }
        
        actions_tensor = torch.LongTensor(actions_batch).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs_batch).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages_batch).to(self.device)
        returns_tensor = torch.FloatTensor(returns_batch).to(self.device)
        
        # ENHANCED: Normalize advantages for better training stability
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update with ENHANCED loss computation
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        for epoch in range(self.ppo_epochs):
            # Forward pass
            policy_family_logits, scope_logits, target_logits, strength_logits = self.policy_net(obs_tensor)
            values = self.value_net(obs_tensor)
            
            # Compute new log probabilities
            policy_family_dist = Categorical(logits=policy_family_logits)
            scope_dist = Categorical(logits=scope_logits)
            target_dist = Categorical(logits=target_logits)
            strength_dist = Categorical(logits=strength_logits)
            
            new_log_probs = (
                policy_family_dist.log_prob(actions_tensor[:, 0]) +
                scope_dist.log_prob(actions_tensor[:, 1]) +
                target_dist.log_prob(actions_tensor[:, 2]) +
                strength_dist.log_prob(actions_tensor[:, 3])
            )
            
            # ENHANCED: Compute ratios with better numerical stability
            log_ratio = new_log_probs - old_log_probs_tensor.sum(dim=1)
            ratios = torch.exp(torch.clamp(log_ratio, -20, 20))  # Prevent extreme values
            
            # Policy loss with ENHANCED clipping
            surr1 = ratios * advantages_tensor
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # ENHANCED: Value loss with clipping for better stability
            value_pred = values.squeeze()
            value_loss_unclipped = F.mse_loss(value_pred, returns_tensor)
            value_clipped = returns_tensor + torch.clamp(
                value_pred - returns_tensor, 
                -self.clip_epsilon, 
                self.clip_epsilon
            )
            value_loss_clipped = F.mse_loss(value_clipped, returns_tensor)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
            
            # ENHANCED: Entropy loss with better scaling
            entropy = (
                policy_family_dist.entropy().mean() +
                scope_dist.entropy().mean() +
                target_dist.entropy().mean() +
                strength_dist.entropy().mean()
            )
            entropy_loss = -entropy
            
            # ENHANCED: Total loss with better coefficients
            loss = (
                policy_loss + 
                self.value_loss_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            # Backward pass with ENHANCED gradient handling
            self.optimizer.zero_grad()
            loss.backward()
            
            # ENHANCED: Separate gradient clipping for policy and value networks
            policy_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            value_norm = torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            policy_loss += policy_loss.item()
            value_loss += value_loss.item()
            entropy_loss += entropy_loss.item()
        
        # ENHANCED: Return average losses and additional metrics
        avg_losses = {
            'total_loss': total_loss / self.ppo_epochs,
            'policy_loss': policy_loss / self.ppo_epochs,
            'value_loss': value_loss / self.ppo_epochs,
            'entropy_loss': entropy_loss / self.ppo_epochs,
            'policy_grad_norm': policy_norm.item(),
            'value_grad_norm': value_norm.item(),
        }
        
        # Store losses for monitoring
        self.training_losses.append(avg_losses['total_loss'])
        self.policy_losses.append(avg_losses['policy_loss'])
        self.value_losses.append(avg_losses['value_loss'])
        self.entropy_losses.append(avg_losses['entropy_loss'])
        
        return avg_losses
    
    def train(self, num_episodes: int, checkpoint_frequency: int = 100) -> None:
        """Main training loop."""
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Progress log: {self.logger.log_file}")
        print(f"Environment years: {self.env.years} (optimized for speed)")
        print(f"Batch size: {self.batch_size}, PPO epochs: {self.ppo_epochs}")
        print("-" * 80)
        
        start_time = time.time()
        episode_times = []
        
        for episode in range(self.episode_count, self.episode_count + num_episodes):
            episode_start_time = time.time()
            
            # Show progress bar
            current_episode = episode - self.episode_count + 1
            print_progress_bar(
                current_episode, num_episodes,
                prefix=f'Training Progress',
                suffix=f'Episode {episode:06d} | Best: {self.best_reward:7.3f}',
                length=40
            )
            
            # Collect episode
            episode_data, total_reward, step_count = self.collect_episode()
            
            # Update policy
            if len(episode_data) > 0:
                update_info = self.update_policy(episode_data)
            else:
                update_info = {}
            
            # Update tracking
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step_count)
            self.total_steps += step_count
            
            # ENHANCED: Check if this is the best episode and track improvement
            is_best = total_reward > self.best_reward
            if is_best:
                self.best_reward = total_reward
                self.episodes_without_improvement = 0
                print(f"\n🎉 NEW BEST REWARD: {self.best_reward:.3f} at episode {episode}")
            else:
                self.episodes_without_improvement += 1
            
            # ENHANCED: Early stopping if no improvement for too long
            if self.episodes_without_improvement >= self.early_stopping_patience:
                print(f"\n⚠️  EARLY STOPPING: No improvement for {self.early_stopping_patience} episodes")
                print(f"   Best reward achieved: {self.best_reward:.3f}")
                break
            
            # Calculate episode time
            episode_time = time.time() - episode_start_time
            episode_times.append(episode_time)
            
            # Calculate average episode time
            avg_episode_time = np.mean(episode_times[-10:])  # Last 10 episodes
            estimated_remaining = avg_episode_time * (num_episodes - current_episode)
            
            # ENHANCED: Calculate more detailed statistics
            avg_reward = np.mean(list(self.episode_rewards))
            recent_avg_reward = np.mean(list(self.episode_rewards)[-50:]) if len(self.episode_rewards) >= 50 else avg_reward
            
            # ENHANCED: Learning rate scheduling based on performance
            if episode % 100 == 0 and episode > 0:
                self.scheduler.step(recent_avg_reward)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\n📊 Learning Rate: {current_lr:.2e} | Recent Avg Reward: {recent_avg_reward:.3f}")
            
            # Clear progress bar and show ENHANCED episode summary
            print(f"\rEpisode {episode:06d} | "
                  f"Reward: {total_reward:7.3f} | "
                  f"Avg: {avg_reward:7.3f} | "
                  f"Recent: {recent_avg_reward:7.3f} | "
                  f"Best: {self.best_reward:7.3f} | "
                  f"Steps: {step_count:3d} | "
                  f"Time: {episode_time:5.1f}s | "
                  f"ETA: {estimated_remaining/60:5.1f}min | "
                  f"No Imp: {self.episodes_without_improvement:3d}")
            
            # ENHANCED: Log detailed progress information
            self.logger.log_episode(
                episode, total_reward, avg_reward, self.best_reward, 
                step_count, update_info, episode_time
            )
            
            # ENHANCED: Checkpoint with better frequency
            if episode % checkpoint_frequency == 0 or is_best:
                self.save_checkpoint(episode, is_best)
                if is_best:
                    print(f"💾 Best model checkpoint saved!")
            
            # ENHANCED: Performance monitoring every 100 episodes
            if episode % 100 == 0 and episode > 0:
                self._print_performance_summary()
            
            # Update episode count
            self.episode_count = episode + 1
        
        # Final checkpoint
        self.save_checkpoint(self.episode_count - 1, True)
        
        training_time = time.time() - start_time
        final_avg_reward = np.mean(list(self.episode_rewards))
        avg_episode_time = np.mean(episode_times)
        
        # ENHANCED: Calculate final performance statistics
        final_recent_avg = np.mean(list(self.episode_rewards)[-100:]) if len(self.episode_rewards) >= 100 else final_avg_reward
        reward_std = np.std(list(self.episode_rewards))
        improvement_rate = (self.best_reward - list(self.episode_rewards)[0]) / len(self.episode_rewards) if len(self.episode_rewards) > 1 else 0
        
        print(f"\n" + "="*80)
        print(f"🎯 ENHANCED TRAINING COMPLETED!")
        print(f"="*80)
        print(f"Total Episodes: {num_episodes}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Best Reward: {self.best_reward:.3f}")
        print(f"Final Average Reward: {final_avg_reward:.3f}")
        print(f"Recent Average Reward (last 100): {final_recent_avg:.3f}")
        print(f"Reward Standard Deviation: {reward_std:.3f}")
        print(f"Improvement Rate per Episode: {improvement_rate:.6f}")
        print(f"Average Episode Time: {avg_episode_time:.2f} seconds")
        print(f"Total Training Time: {training_time/3600:.2f} hours")
        print(f"Training Speed: {num_episodes/training_time*3600:.1f} episodes/hour")
        print(f"Final Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"="*80)
        
        # ENHANCED: Log final summary with more details
        self.logger.log_summary(
            self.episode_count, self.total_steps, self.best_reward,
            final_avg_reward, training_time
        )
    
    def _print_performance_summary(self):
        """Print detailed performance summary every 100 episodes."""
        if len(self.episode_rewards) < 10:
            return
        
        recent_rewards = list(self.episode_rewards)[-100:]
        recent_lengths = list(self.episode_lengths)[-100:]
        
        avg_recent_reward = np.mean(recent_rewards)
        std_recent_reward = np.std(recent_rewards)
        avg_recent_length = np.mean(recent_lengths)
        
        if len(self.training_losses) > 0:
            recent_losses = list(self.training_losses)[-50:]
            avg_recent_loss = np.mean(recent_losses)
            print(f"\n📊 PERFORMANCE SUMMARY (Last 100 episodes):")
            print(f"   Average Reward: {avg_recent_reward:.3f} ± {std_recent_reward:.3f}")
            print(f"   Average Episode Length: {avg_recent_length:.1f} steps")
            print(f"   Average Training Loss: {avg_recent_loss:.4f}")
            print(f"   Best Reward So Far: {self.best_reward:.3f}")
            print(f"   Episodes Without Improvement: {self.episodes_without_improvement}")
            print(f"   Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")


def main():
    """Main training function with IMPROVED settings for better RL performance."""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Check GPU availability first
    gpu_status = check_gpu_availability()
    
    # Create environment with ENHANCED settings for better learning
    print("\nCreating Turkey Policy Environment with ENHANCED settings...")
    env = TurkeyPolicyEnv(
        years=30,  # INCREASED from 25 to 30 for better policy learning
        seed=42,
        reward_weights={
            "w_gini": 1.2,      # INCREASED: More emphasis on inequality reduction
            "w_gdp": 0.8,       # INCREASED: Better GDP growth focus
            "w_cost": 0.15,     # REDUCED: Less penalty for policy costs
            "w_vol": 0.08,      # INCREASED: More emphasis on stability
            "w_nw": 0.12,       # INCREASED: Better nationwide policy balance
            "w_floor": 0.4,     # INCREASED: More emphasis on lifting bottom cities
        },
        # Migration weights for open economy models
        gdp_weight=1.0,
        diaspora_weight=1.0,
        distance_weight=-1.4,  # NEGATIVE: closer cities attract more migration
        source_pop_weight=1.0,
        target_pop_weight=1.0,
    )
    
    print(f"ENHANCED Migration weights configured:")
    print(f"  GDP weight: {env.gdp_weight}")
    print(f"  Diaspora weight: {env.diaspora_weight}")
    print(f"  Distance weight: {env.distance_weight} (negative = closer cities attract more)")
    print(f"  Source population weight: {env.source_pop_weight}")
    print(f"  Target population weight: {env.target_pop_weight}")
    
    print(f"\nENHANCED Reward weights configured:")
    print(f"  Gini coefficient weight: {env.reward_weights.w_gini} (inequality reduction)")
    print(f"  GDP per capita weight: {env.reward_weights.w_gdp} (economic growth)")
    print(f"  Policy cost weight: {env.reward_weights.w_cost} (cost efficiency)")
    print(f"  Volatility weight: {env.reward_weights.w_vol} (stability)")
    print(f"  Nationwide overuse weight: {env.reward_weights.w_nw} (policy balance)")
    print(f"  Bottom-lifting weight: {env.reward_weights.w_floor} (equity)")
    
    # Create trainer with ENHANCED hyperparameters for better learning
    print("\nCreating PPO Trainer with ENHANCED hyperparameters...")
    trainer = PPOTrainer(
        env=env,
        learning_rate=3e-4,      # OPTIMIZED: Balanced learning rate
        gamma=0.985,             # INCREASED: Better long-term planning
        gae_lambda=0.97,         # INCREASED: Better advantage estimation
        clip_epsilon=0.15,       # REDUCED: More conservative policy updates
        value_loss_coef=0.6,     # INCREASED: Better value function learning
        entropy_coef=0.02,       # OPTIMIZED: Balanced exploration vs exploitation
        max_grad_norm=0.3,       # REDUCED: More stable training
        ppo_epochs=3,            # OPTIMIZED: Balanced between speed and quality
        batch_size=48,           # INCREASED: Better gradient estimates
        checkpoint_dir="checkpoints",
        device="auto",
    )
    
    # Show final device confirmation
    print(f"\nFinal device being used: {trainer.device}")
    if str(trainer.device).startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(trainer.device)}")
        print(f"Memory: {torch.cuda.get_device_properties(trainer.device).total_memory / 1024**3:.1f} GB")
    
    # Show ENHANCED training configuration
    print(f"\n🚀 ENHANCED TRAINING CONFIGURATION:")
    print(f"   Episodes: 3000 (increased for better learning)")
    print(f"   Episode Length: 30 years (increased for policy complexity)")
    print(f"   Learning Rate: 3e-4 (optimized for stability)")
    print(f"   Gamma: 0.985 (better long-term planning)")
    print(f"   GAE Lambda: 0.97 (better advantage estimation)")
    print(f"   Clip Epsilon: 0.15 (more conservative updates)")
    print(f"   Value Loss Coef: 0.6 (better value learning)")
    print(f"   Entropy Coef: 0.02 (balanced exploration)")
    print(f"   Max Grad Norm: 0.3 (more stable training)")
    print(f"   PPO Epochs: 3 (balanced quality/speed)")
    print(f"   Batch Size: 48 (better gradient estimates)")
    
    # Start training with ENHANCED settings
    print("\n🎯 Starting ENHANCED training for better RL performance...")
    trainer.train(
        num_episodes=3000,       # INCREASED: More episodes for better learning
        checkpoint_frequency=100, # OPTIMIZED: Better checkpoint balance
    )


if __name__ == "__main__":
    main() 