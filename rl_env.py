import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Tuple
from dataclasses import dataclass

# Import TReconomy only when needed to avoid circular imports
# from TReconomy import TReconomy


@dataclass
class RewardWeights:
    """Reward function weights for the RL agent."""
    w_gini: float = 1.0      # Gini coefficient improvement weight
    w_gdp: float = 0.7       # GDP per capita growth weight  
    w_cost: float = 0.2      # Policy cost penalty weight
    w_vol: float = 0.05      # Policy volatility penalty weight
    w_nw: float = 0.1        # Nationwide overuse penalty weight
    w_floor: float = 0.3     # Bottom-lifting bonus weight


class TurkeyPolicyEnv:
    """
    Reinforcement Learning environment that wraps TReconomy.
    Provides gym-like interface for policy training.
    
    The agent observes the economy state and chooses policies to improve economic outcomes.
    NO OUTPUT FILES - Pure training only.
    """
    
    def __init__(
        self,
        years: int = 100,
        seed: int = 42,
        reward_weights: Optional[Dict[str, float]] = None,
        map_data_path: str = "harita_dosyaları",
        nodes_file: str = "datalar/network_nodes.xlsx",
        edges_file: str = "datalar/network_edge_weights.xlsx",
        current_year: int = 2024,
        # Migration weights for open economy models
        gdp_weight: float = 1.0,
        diaspora_weight: float = 1.0,
        distance_weight: float = -1.4,
        source_pop_weight: float = 1.0,
        target_pop_weight: float = 1.0,
    ):
        self.years = years
        self.seed = seed
        self.current_year = current_year
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Reward weights
        if reward_weights is None:
            reward_weights = {}
        self.reward_weights = RewardWeights(**reward_weights)
        
        # Migration weights
        self.gdp_weight = gdp_weight
        self.diaspora_weight = diaspora_weight
        self.distance_weight = distance_weight
        self.source_pop_weight = source_pop_weight
        self.target_pop_weight = target_pop_weight
        
        # Initialize economy (will be created in reset())
        self.economy = None
        self.current_year_idx = 0
        self.episode_data = []
        
        # Store paths for economy creation
        self.map_data_path = map_data_path
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        
        # Action space: MultiDiscrete([6, 3, 81, 4])
        # - 6 policy families
        # - 3 scopes (nationwide, regional, city)
        # - 81 cities (target)
        # - 4 strength levels (0=no-op, 1=low, 2=mid, 3=high)
        self.action_space = (6, 3, 81, 4)
        
        # Observation space
        # - 81 cities × 13 features per city
        # - 3 global features
        self.observation_space = {
            'node_features': (81, 13),  # cities × features
            'global_features': (3,),     # global state
        }
        
        # Policy cost mapping (strength level → cost)
        self.policy_costs = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.6}
        
        # Track previous action for volatility penalty
        self.previous_action = None
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment for new episode.
        Creates fresh TReconomy instance and returns initial observation.
        """
        # Import TReconomy locally to avoid circular imports
        from TReconomy import TReconomy
        
        # Create new economy instance for this episode
        self.economy = TReconomy(
            map_data_path=self.map_data_path,
            nodes_file=self.nodes_file,
            edges_file=self.edges_file,
            current_year=self.current_year,
            years=self.years,
            model_type="policy",  # Use policy model for RL
            migration_rate=0.01,
            capital_migration_ratio=0.2,
            population_growth_rate=0.013,
            labor_growth_rate=0.013,
            # Pass migration weights
            gdp_weight=self.gdp_weight,
            diaspora_weight=self.diaspora_weight,
            distance_weight=self.distance_weight,
            source_pop_weight=self.source_pop_weight,
            target_pop_weight=self.target_pop_weight,
        )
        
        # CRITICAL: Disable all output generation
        self._disable_outputs()
        
        # Reset episode state
        self.current_year_idx = 0
        self.episode_data = []
        self.previous_action = None
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs
    
    def _disable_outputs(self):
        """Disable all output generation methods to prevent file creation."""
        if self.economy is None:
            return
        
        # Override methods that create outputs
        def no_export(*args, **kwargs):
            pass  # Do nothing
        
        def no_animate(*args, **kwargs):
            pass  # Do nothing
        
        def no_plot(*args, **kwargs):
            pass  # Do nothing
        
        # Disable all output methods
        self.economy.export_results = no_export
        self.economy.export_shock_results = no_export
        self.economy.export_detailed_shock_results = no_export
        self.economy.export_policy_results = no_export
        self.economy.export_detailed_policy_results = no_export
        self.economy.export_model_comparison = no_export
        self.economy.animate_gdp_per_capita = no_animate
        self.economy.animate_gdp_per_capita_enhanced = no_animate
        self.economy.animate_model_comparison = no_animate
        self.economy.animate_shock_effects = no_animate
        self.economy.animate_policy_effects = no_animate
        self.economy.plot_gini_comparison = no_plot
        self.economy.plot_individual_gini = no_plot
        self.economy.plot_combined_gini = no_plot
    
    def step(self, action: Tuple[int, int, int, int]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Take action in the environment.
        
        Args:
            action: Tuple of (policy_family, scope, target_city, strength)
            
        Returns:
            observation: Current state observation
            reward: Reward for this action
            done: Whether episode is finished
            info: Additional information
        """
        if self.economy is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Parse action
        policy_family, scope, target_city, strength = action
        
        # Apply policy if strength > 0 (no-op if strength = 0)
        if strength > 0:
            self._apply_policy(policy_family, scope, target_city, strength)
        
        # Advance simulation by one year
        self._simulate_year()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is done
        done = self.current_year_idx >= self.years
        
        # OPTIMIZATION: Early termination for better performance
        if not done:
            # Early termination if reward is consistently good
            if len(self.episode_data) >= 5:
                recent_rewards = [step['reward'] for step in self.episode_data[-5:]]
                if all(r > 0.5 for r in recent_rewards):  # Good performance threshold
                    done = True
                    reward += 2.0  # Bonus for early success
            
            # Early termination if reward is consistently bad (avoid wasting time)
            elif len(self.episode_data) >= 10:
                recent_rewards = [step['reward'] for step in self.episode_data[-10:]]
                if all(r < -0.5 for r in recent_rewards):  # Poor performance threshold
                    done = True
                    reward -= 1.0  # Penalty for poor performance

        # Store episode data
        self.episode_data.append({
            'year': self.current_year_idx,
            'action': action,
            'reward': reward,
            'gini': self._get_current_gini(),
            'gdp_pc': self._get_current_gdp_per_capita(),
        })
        
        # Update previous action for next step
        self.previous_action = action
        
        # Info dictionary
        info = {
            'year': self.current_year_idx,
            'policy_applied': strength > 0,
            'gini': self._get_current_gini(),
            'gdp_pc': self._get_current_gdp_per_capita(),
            'early_termination': done and self.current_year_idx < self.years,
        }
        
        return obs, reward, done, info
    
    def _apply_policy(self, policy_family: int, scope: int, target_city: int, strength: int):
        """
        Apply policy to the economy based on action.
        This modifies the TReconomy instance directly.
        """
        # Convert indices to actual values
        policy_families = [
            "Economic Growth", "Health/Education", "Infrastructure Development",
            "Technology & Innovation", "Environmental", "Social/Demographic"
        ]
        scopes = ["nationwide", "regional_k5", "city"]
        
        policy_family_name = policy_families[policy_family]
        scope_name = scopes[scope]
        
        # Get city name from index
        city_names = list(self.economy.tr_network.cities.keys())
        target_city_name = city_names[target_city]
        
        # Create policy effect based on family and strength
        effect_multiplier = strength * 0.1  # Base effect: 0.1, 0.2, 0.3 for strengths 1,2,3
        
        # Policy effects vary by family
        if policy_family == 0:  # Economic Growth
            effects = {
                "A": 0.05 * effect_multiplier,
                "K": 0.03 * effect_multiplier,
                "L": 0.02 * effect_multiplier,
                "ALPHA": 0.02 * effect_multiplier,
                "BETA": 0.02 * effect_multiplier,
            }
        elif policy_family == 1:  # Health/Education
            effects = {
                "A": 0.04 * effect_multiplier,
                "K": 0.02 * effect_multiplier,
                "L": 0.06 * effect_multiplier,
                "ALPHA": 0.01 * effect_multiplier,
                "BETA": 0.05 * effect_multiplier,
            }
        elif policy_family == 2:  # Infrastructure
            effects = {
                "A": 0.06 * effect_multiplier,
                "K": 0.08 * effect_multiplier,
                "L": 0.03 * effect_multiplier,
                "ALPHA": 0.04 * effect_multiplier,
                "BETA": 0.03 * effect_multiplier,
            }
        elif policy_family == 3:  # Technology
            effects = {
                "A": 0.08 * effect_multiplier,
                "K": 0.05 * effect_multiplier,
                "L": 0.02 * effect_multiplier,
                "ALPHA": 0.06 * effect_multiplier,
                "BETA": 0.04 * effect_multiplier,
            }
        elif policy_family == 4:  # Environmental
            effects = {
                "A": 0.03 * effect_multiplier,
                "K": 0.06 * effect_multiplier,
                "L": 0.02 * effect_multiplier,
                "ALPHA": 0.03 * effect_multiplier,
                "BETA": 0.02 * effect_multiplier,
            }
        else:  # Social/Demographic
            effects = {
                "A": 0.04 * effect_multiplier,
                "K": 0.03 * effect_multiplier,
                "L": 0.05 * effect_multiplier,
                "ALPHA": 0.02 * effect_multiplier,
                "BETA": 0.04 * effect_multiplier,
            }
        
        # Apply effects based on scope
        if scope_name == "nationwide":
            # Apply to all cities
            for city in self.economy.tr_network.cities.values():
                self._apply_city_effects(city, effects)
        elif scope_name == "regional_k5":
            # Apply to target city and 5 nearest neighbors
            target_city_obj = self.economy.tr_network.cities[target_city_name]
            nearby_cities = self._get_nearest_cities(target_city_obj, 5)
            for city in nearby_cities:
                self._apply_city_effects(city, effects)
        else:  # city
            # Apply only to target city
            target_city_obj = self.economy.tr_network.cities[target_city_name]
            self._apply_city_effects(target_city_obj, effects)
    
    def _apply_city_effects(self, city, effects: Dict[str, float]):
        """Apply policy effects to a city's parameters."""
        for param, effect in effects.items():
            if param == "A":
                city.A *= (1 + effect)
            elif param == "K":
                city.capital_stock *= (1 + effect)
            elif param == "L":
                city.labor_force = int(city.labor_force * (1 + effect))
            elif param == "ALPHA":
                city.alpha *= (1 + effect)
            elif param == "BETA":
                city.beta *= (1 + effect)
    
    def _get_nearest_cities(self, target_city, num_neighbors: int) -> List:
        """Get nearest neighbor cities for regional policies."""
        cities = list(self.economy.tr_network.cities.values())
        distances = [(city, target_city.distance_to(city)) for city in cities if city != target_city]
        distances.sort(key=lambda x: x[1])
        return [city for city, _ in distances[:num_neighbors]]
    
    def _simulate_year(self):
        """Simulate one year of the economy."""
        if self.economy is None:
            return
        
        # Trigger shocks for this year (if any)
        if hasattr(self.economy, 'trigger_random_shocks'):
            self.economy.trigger_random_shocks(self.current_year + self.current_year_idx)
        
        # Apply migration and update economy
        if self.current_year_idx == 0:
            # First year: just initialize
            pass
        else:
            # Subsequent years: apply migration and growth
            if self.economy.model_type in ["open", "shock", "policy"]:
                migration_flows = self.economy.calculate_migration_populations(
                    self.current_year + self.current_year_idx
                )
                if migration_flows:
                    self.economy.apply_migration(migration_flows)
        
        # Update city parameters for next year
        for city in self.economy.tr_network.cities.values():
            # Population and labor growth
            city.population = int(city.population * (1 + self.economy.population_growth_rate))
            city.labor_force = int(city.labor_force * (1 + self.economy.labor_growth_rate))
            
            # Maintain demographic proportions
            self.economy._maintain_demographic_proportions(city)
            
            # Capital accumulation
            production = city.calculate_production()
            city.capital_stock += 0.05 * production
        
        # Update year counter
        self.current_year_idx += 1
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation of the economy state."""
        if self.economy is None:
            return self._get_empty_observation()
        
        # Node features: per-city information
        node_features = []
        for city in self.economy.tr_network.cities.values():
            try:
                gdp_pc = city.calculate_gdp_per_capita()
            except:
                gdp_pc = 0.0
            
            # Calculate migration flows (in/out, normalized)
            migration_in = sum(m['Migration_Population'] for m in self.economy.migration_results 
                             if m['Target_City'] == city.name) if hasattr(self.economy, 'migration_results') else 0
            migration_out = sum(m['Migration_Population'] for m in self.economy.migration_results 
                              if m['Source_City'] == city.name) if hasattr(self.economy, 'migration_results') else 0
            
            # Normalize migration by city population
            pop_norm = max(city.population, 1)
            migration_in_norm = migration_in / pop_norm
            migration_out_norm = migration_out / pop_norm
            
            # Capital migration (normalized)
            capital_in = sum(m.get('Migration_Capital', 0) for m in self.economy.migration_results 
                           if m['Target_City'] == city.name) if hasattr(self.economy, 'migration_results') else 0
            capital_out = sum(m.get('Migration_Capital', 0) for m in self.economy.migration_results 
                            if m['Source_City'] == city.name) if hasattr(self.economy, 'migration_results') else 0
            
            capital_norm = max(city.capital_stock, 1)
            capital_in_norm = capital_in / capital_norm
            capital_out_norm = capital_out / capital_norm
            
            # City features vector (13 features)
            city_features = [
                city.A,                    # Total Factor Productivity
                city.capital_stock,        # Capital stock
                city.labor_force,          # Labor force
                city.alpha,                # Capital share parameter
                city.beta,                 # Labor share parameter
                city.population,           # Population
                gdp_pc,                   # GDP per capita
                migration_in_norm,         # Normalized migration in
                migration_out_norm,        # Normalized migration out
                capital_in_norm,           # Normalized capital in
                capital_out_norm,          # Normalized capital out
                city.labor_force / max(city.population, 1),  # Labor participation rate
                city.capital_stock / max(city.labor_force, 1),  # Capital-labor ratio
            ]
            node_features.append(city_features)
        
        # Global features: economy-wide information
        total_pop = sum(city.population for city in self.economy.tr_network.cities.values())
        total_production = sum(city.calculate_production() for city in self.economy.tr_network.cities.values())
        current_gini = self._get_current_gini()
        
        global_features = [
            self.current_year_idx / self.years,  # Progress through episode (0 to 1)
            current_gini,                        # Current Gini coefficient
            total_production / max(total_pop, 1),  # Average GDP per capita
        ]
        
        return {
            'node_features': np.array(node_features, dtype=np.float32),
            'global_features': np.array(global_features, dtype=np.float32),
        }
    
    def _get_empty_observation(self) -> Dict[str, np.ndarray]:
        """Return empty observation when economy not initialized."""
        return {
            'node_features': np.zeros((81, 13), dtype=np.float32),
            'global_features': np.zeros(3, dtype=np.float32),
        }
    
    def _get_current_gini(self) -> float:
        """Get current Gini coefficient."""
        if self.economy is None:
            return 0.0
        
        gdp_values = []
        for city in self.economy.tr_network.cities.values():
            try:
                gdp_pc = city.calculate_gdp_per_capita()
                if gdp_pc > 0:
                    gdp_values.append(gdp_pc)
            except:
                continue
        
        if len(gdp_values) < 2:
            return 0.0
        
        return self._calculate_gini(gdp_values)
    
    def _get_current_gdp_per_capita(self) -> float:
        """Get current average GDP per capita."""
        if self.economy is None:
            return 0.0
        
        total_gdp = 0
        total_pop = 0
        for city in self.economy.tr_network.cities.values():
            try:
                gdp = city.calculate_production()
                total_gdp += gdp
                total_pop += city.population
            except:
                continue
        
        return total_gdp / max(total_pop, 1)
    
    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values."""
        if len(values) < 2:
            return 0.0
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumvals = [sum(sorted_vals[:i+1]) for i in range(n)]
        gini = (n + 1 - 2 * sum((i + 1) * v for i, v in enumerate(sorted_vals)) / sum(sorted_vals)) / n
        return abs(gini)
    
    def _calculate_reward(self, action: Tuple[int, int, int, int]) -> float:
        """ENHANCED reward function with better reward shaping and multi-objective optimization."""
        if self.economy is None or len(self.episode_data) < 1:
            return 0.0
        
        policy_family, scope, target_city, strength = action
        
        # Base reward components
        reward = 0.0
        
        # ENHANCED: 1. Gini coefficient improvement with better normalization
        if len(self.episode_data) > 1:
            prev_gini = self.episode_data[-2]['gini']
            curr_gini = self.episode_data[-1]['gini']
            
            # Better normalization: consider both absolute and relative improvement
            gini_improvement = (prev_gini - curr_gini) / max(prev_gini, 0.01)  # Relative improvement
            gini_improvement_abs = (prev_gini - curr_gini) / 0.01  # Absolute improvement
            
            # Combine both metrics for better reward signal
            gini_reward = 0.7 * gini_improvement + 0.3 * gini_improvement_abs
            reward += self.reward_weights.w_gini * gini_reward
            
            # Bonus for significant inequality reduction
            if gini_improvement > 0.1:  # 10% improvement
                reward += self.reward_weights.w_gini * 0.5
        
        # ENHANCED: 2. GDP per capita growth with better metrics
        if len(self.episode_data) > 1:
            prev_gdp = self.episode_data[-2]['gdp_pc']
            curr_gdp = self.episode_data[-1]['gdp_pc']
            
            if prev_gdp > 0:
                # Log growth for better numerical stability
                gdp_growth = (np.log(curr_gdp) - np.log(prev_gdp)) / 0.02
                reward += self.reward_weights.w_gdp * gdp_growth
                
                # Bonus for sustained growth
                if len(self.episode_data) > 2:
                    prev_prev_gdp = self.episode_data[-3]['gdp_pc']
                    if prev_prev_gdp > 0:
                        sustained_growth = (np.log(curr_gdp) - np.log(prev_prev_gdp)) / 0.04
                        reward += self.reward_weights.w_gdp * 0.3 * sustained_growth
        
        # ENHANCED: 3. Policy cost penalty with diminishing returns
        policy_cost = self.policy_costs[strength]
        # Use square root penalty for better cost-benefit balance
        cost_penalty = self.reward_weights.w_cost * np.sqrt(policy_cost)
        reward -= cost_penalty
        
        # ENHANCED: 4. Policy volatility penalty with adaptive scaling
        if self.previous_action is not None:
            action_diff = np.array(action) - np.array(self.previous_action)
            volatility = np.linalg.norm(action_diff)
            
            # Adaptive penalty: more penalty for drastic changes
            if volatility > 3:
                volatility_penalty = self.reward_weights.w_vol * (volatility ** 1.5)
            else:
                volatility_penalty = self.reward_weights.w_vol * volatility
            
            reward -= volatility_penalty
        
        # ENHANCED: 5. Nationwide overuse penalty with smart balancing
        if scope == 0:  # nationwide
            # Count nationwide policies in recent history
            recent_nationwide = sum(1 for step in self.episode_data[-5:] if step['action'][1] == 0)
            if recent_nationwide > 2:  # Too many nationwide policies
                reward -= self.reward_weights.w_nw * (recent_nationwide - 2)
            else:
                # Small bonus for strategic nationwide use
                reward += self.reward_weights.w_nw * 0.1
        
        # ENHANCED: 6. Bottom-lifting bonus with better targeting
        if len(self.episode_data) > 1:
            cities = list(self.economy.tr_network.cities.values())
            gdp_values = [city.calculate_gdp_per_capita() for city in cities]
            
            # Check bottom 20% improvement
            sorted_gdp = sorted(gdp_values)
            bottom_20_idx = max(1, len(sorted_gdp) // 5)
            bottom_20_avg = np.mean(sorted_gdp[:bottom_20_idx])
            
            prev_bottom_20 = min(self.episode_data[-2]['gdp_pc'], 1e-6)
            if prev_bottom_20 > 0:
                bottom_improvement = (bottom_20_avg - prev_bottom_20) / max(prev_bottom_20, 0.01)
                reward += self.reward_weights.w_floor * bottom_improvement
        
        # ENHANCED: 7. Policy effectiveness bonus
        if len(self.episode_data) > 1:
            # Check if the policy actually improved the target metric
            if policy_family in [0, 2, 3]:  # Growth, Infrastructure, Technology
                # These should improve GDP
                if curr_gdp > prev_gdp:
                    reward += 0.1  # Small bonus for effective policy
            elif policy_family in [4, 5]:  # Environmental, Social
                # These should improve Gini
                if curr_gini < prev_gini:
                    reward += 0.1  # Small bonus for effective policy
        
        # ENHANCED: 8. Exploration bonus for trying new combinations
        if len(self.episode_data) > 5:
            recent_actions = [step['action'] for step in self.episode_data[-5:]]
            action_diversity = len(set(str(action) for action in recent_actions))
            if action_diversity >= 4:  # Good diversity
                reward += 0.05  # Small exploration bonus
        
        # ENHANCED: 9. Stability bonus for consistent performance
        if len(self.episode_data) > 3:
            recent_rewards = [step['reward'] for step in self.episode_data[-3:]]
            if all(r > 0 for r in recent_rewards):  # Consistent positive rewards
                reward += 0.1  # Stability bonus
        
        # ENHANCED: 10. Final reward clipping and scaling
        reward = np.clip(reward, -10.0, 10.0)  # Prevent extreme values
        
        return reward
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the completed episode."""
        if not self.episode_data:
            return {}
        
        total_reward = sum(step['reward'] for step in self.episode_data)
        final_gini = self.episode_data[-1]['gini']
        final_gdp = self.episode_data[-1]['gdp_pc']
        
        return {
            'total_reward': total_reward,
            'final_gini': final_gini,
            'final_gdp_pc': final_gdp,
            'episode_length': len(self.episode_data),
            'policy_count': sum(1 for step in self.episode_data if step['action'][3] > 0),
        } 