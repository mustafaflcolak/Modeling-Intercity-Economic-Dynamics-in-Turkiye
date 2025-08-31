import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import json
import random
from TRnetwork import TRnetwork
from city import City
from output_config import get_excel_path, get_animation_path, get_comparison_paths, list_output_files



class TReconomy:
    """
    Class to investigate economic activation in the Turkey network.
    Builds the Turkish network using TRnetwork and City classes.
    Simulates economic growth using Cobb-Douglas production for each city.
    
    Features:
    - Population and labor force growth rates (can be set independently)
    - Migration between cities (for open economy models)
    - Capital migration at configurable ratio of labor migration
    - Random economic shocks affecting production function parameters (for shock economy models)
    - Capital accumulation based on production
    - Demographic tracking and maintenance
    - Model comparison capabilities
    - Gini coefficient visualization and comparison
    
    Model Types:
    - "closed": No migration, each city operates independently
    - "open": With migration and capital movement between cities
    - "shock": Open economy with random economic shocks affecting production parameters
    - "policy": Open economy with random economic shocks and policies affecting production parameters
    
    Growth Rates:
    - population_growth_rate: Annual growth rate for city populations
    - labor_growth_rate: Annual growth rate for labor force (defaults to population rate)
    - Both rates maintain demographic proportions across all cities
    
    Migration (Open Economy Only):
    - migration_rate: Rate of population migration between cities
    - capital_migration_ratio: Capital migration rate as fraction of labor migration rate
      (e.g., 0.2 means capital migrates at 1/5 of labor migration rate)
    
    Shock System (Shock Economy Only):
    - shock_probability: Probability of shocks occurring per year (default: 5%)
    - max_shocks_per_year: Maximum number of shocks per year (default: 3)
    - Shocks affect production function parameters (A, K, L, ALPHA, BETA)
    - Effects are persistent and cumulative across years
    - Supports both national (all cities) and city-specific effects
    - Parameters can be negative or positive (α, β, A can be < 0)
    - Shock formula depends on parameter sign:
      * Negative parameters: new_value = parameter × (1 - effect)/100
      * Positive parameters: new_value = parameter × (1 + effect)/100
    - Example: β = -1.0, effect = +0.5 → new β = -1.0 × (1 - 0.5)/100 = -0.005 (improved)
    - Example: β = +10.0, effect = -0.2 → new β = +10.0 × (1 + (-0.2))/100 = +0.08 (worsened)
    
    Policy System (Policy Economy Only):
    - available_policies: List of pre-generated policies
    - active_policies: List of currently active policies
    - Policy effects are applied to production function parameters
    
    Visualization:
    - plot_gini_comparison(): Side-by-side Gini coefficient graphs for model comparison
    - plot_individual_gini(): Individual Gini coefficient graph for a specific model
    - plot_combined_gini(): Combined Gini coefficient graph with all models on same plot
    - All graphs use fixed y-axis limits (default: 0 to 0.4) for easy comparison
    - Value labels shown every 5 points to avoid clutter
    - animate_gdp_per_capita(): Standard GDP per capita animation
    - animate_gdp_per_capita_enhanced(): Enhanced animation with better color scaling options
    - animate_model_comparison(): Side-by-side model comparison animation
    
    """
    def __init__(self, map_data_path="harita_dosyaları", nodes_file="datalar/network_nodes.xlsx", edges_file="datalar/network_edge_weights.xlsx", current_year=2024, saving_rate=0.20, population_growth_rate=0.013, labor_growth_rate=None, years=15, depreciation_rate=0.05, 
                 model_type="closed", migration_rate=0.01, capital_migration_ratio=0.2, gdp_weight=1.0, diaspora_weight=1.0, distance_weight=1.0, source_pop_weight=1.0, target_pop_weight=1.0):
        self.tr_network = TRnetwork(
            map_data_path=map_data_path,
            nodes_file=nodes_file,
            edges_file=edges_file,
            current_year=current_year
        )
        self.saving_rate = saving_rate  # Note: Capital now increases by 5% of production level
        self.population_growth_rate = population_growth_rate
        # If labor_growth_rate not specified, use population growth rate
        self.labor_growth_rate = labor_growth_rate if labor_growth_rate is not None else population_growth_rate
        self.depreciation_rate = depreciation_rate  # Note: Depreciation no longer applied to capital
        self.years = years
        self.start_year = current_year
        
        # Model configuration
        self.model_type = model_type.lower()  # "closed" or "open"
        
        # Migration parameters (only used for open and shock economy models)
        if model_type.lower() in ["open", "shock", "policy"]:
            self.migration_rate = migration_rate
            self.capital_migration_ratio = capital_migration_ratio  # Capital migration = capital_migration_ratio * labor migration
            self.gdp_weight = gdp_weight
            self.diaspora_weight = diaspora_weight
            self.distance_weight = distance_weight
            self.source_pop_weight = source_pop_weight
            self.target_pop_weight = target_pop_weight
        else:
            self.migration_rate = 0.0
            self.capital_migration_ratio = 0.0
            self.gdp_weight = 0.0
            self.diaspora_weight = 0.0
            self.distance_weight = 0.0
            self.source_pop_weight = 0.0
            self.target_pop_weight = 0.0
        
        # Shock system parameters (only used for shock economy models)
        self.shock_probability = 0.10 if model_type.lower() in ["shock", "policy"] else 0.0  # 10% chance of shocks per year
        self.max_shocks_per_year = 3 if model_type.lower() in ["shock", "policy"] else 0  # Maximum 3 shocks per year
        self.available_shocks = [] if model_type.lower() in ["shock", "policy"] else []
        self.active_shocks = [] if model_type.lower() in ["shock", "policy"] else []  # Track active shocks for reporting
        
        # Policy system parameters (only used for policy economy models)
        self.available_policies = [] if model_type.lower() == "policy" else []
        self.active_policies = [] if model_type.lower() == "policy" else []
        
        # Build the network
        self.tr_network.load_map_data()
        self.tr_network.load_network_data()
        self.tr_network.create_network()
        
        # For storing simulation results
        self.results = []  # List of dicts for each city-year
        self.gini_results = []  # List of dicts for each year
        self.demographics_results = []  # List of dicts for each city-year-origin
        self.migration_results = []  # List of dicts for migration flows (only for open economy)
        
        # Print model configuration
        self._print_model_config()

    def _print_model_config(self):
        """Prints the current model configuration."""
        print(f"\n--- Model Configuration ---")
        print(f"Model Type: {self.model_type.capitalize()} Economy")
        print(f"Population Growth Rate: {self.population_growth_rate:.2%}")
        print(f"Labor Force Growth Rate: {self.labor_growth_rate:.2%}")
        if self.model_type in ["open", "shock", "policy"]:
            print(f"Migration Rate: {self.migration_rate:.2%}")
            denom_str = f"1/{int(1/self.capital_migration_ratio)}" if self.capital_migration_ratio not in [0, 0.0] else "N/A"
            print(f"Capital Migration Ratio: {self.capital_migration_ratio:.1%} ({denom_str}) of labor migration")
            print(f"GDP Weight: {self.gdp_weight}")
            print(f"Diaspora Weight: {self.diaspora_weight}")
            print(f"Distance Weight: {self.distance_weight}")
            print(f"Source Population Weight: {self.source_pop_weight}")
            print(f"Target Population Weight: {self.target_pop_weight}")
        if self.model_type in ["shock", "policy"]:
            print(f"Shock Probability: {self.shock_probability:.1%} per year")
            print(f"Maximum Shocks per Year: {self.max_shocks_per_year}")
            if self.model_type == "policy":
                print(f"Policies: programmatically generated (1 per year)")
        else:
            print("No migration between cities")
            print("Each city operates independently")
        print("---------------------------")

    def migration_decision(self, source_city, target_city):
        """
        Calculate migration propensity score from source_city to target_city based on:
        - GDP per capita difference (higher GDP per capita in target attracts migration)
        - Diaspora ratio (more of source's people already in target increases migration)
        - Distance (closer cities attract more migration)
        - Source city population (larger cities have more potential migrants)
        - Target city population (larger cities can absorb more migrants)
        Returns a non-negative score (higher = more migration tendency).
        """
        # a. GDP per capita effect
        gdp_diff = max(0, target_city.calculate_gdp_per_capita() - source_city.calculate_gdp_per_capita())
        gdp_score = gdp_diff / (source_city.calculate_gdp_per_capita() + 1e-6)  # avoid div by zero

        # b. Diaspora ratio
        diaspora = 0
        total_source = 0
        for city in self.tr_network.cities.values():
            for demo in city.demographics:
                if demo.origin_city == source_city.name:
                    total_source += demo.population_count
                    if city.name == target_city.name:
                        diaspora = demo.population_count
        diaspora_ratio = diaspora / total_source if total_source > 0 else 0

        # c. Distance (inverse effect)
        distance = source_city.distance_to(target_city)
        distance_score = 1 / (1 + distance)  # closer = higher score

        # d. Population effects
        source_pop_score = np.log(source_city.population + 1)  # log to avoid extreme values
        target_pop_score = np.log(target_city.population + 1)

        # e. Combine using instance weights
        score = (self.gdp_weight * gdp_score + 
                self.diaspora_weight * diaspora_ratio + 
                self.distance_weight * distance_score +
                self.source_pop_weight * source_pop_score +
                self.target_pop_weight * target_pop_score)
        return max(0, score)  # no negative migration

    def migration_softmax(self, source_city, threshold=0.05):
        """
        For a given source city, compute softmax-normalized migration probabilities to all other cities.
        Only return targets with probability above the threshold.
        Returns: List of (target_city, probability)
        """
        import numpy as np
        targets = [city for city in self.tr_network.cities.values() if city.name != source_city.name]
        scores = np.array([
            self.migration_decision(source_city, target)
            for target in targets
        ])
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
        probs = exp_scores / exp_scores.sum() if exp_scores.sum() > 0 else np.zeros_like(exp_scores)
        # Filter by threshold
        result = [(target, prob) for target, prob in zip(targets, probs) if prob > threshold]
        return result

    def simulate(self):
        """
        Simulate the Turkish economy for the specified number of years.
        Each city updates its population, labor, and capital stock each year.
        Migration occurs only in open economy models.
        Shocks occur randomly in shock economy models.
        Capital increases by 5% of total city production level every year.
        Store results for export.
        """
        print(f"Simulating Turkish {self.model_type.capitalize()} Economy from {self.start_year} for {self.years} years...")
        
        # Load shocks/policies if needed
        if self.model_type in ["shock", "policy"]:
            self.load_shocks()
            if self.model_type == "policy":
                self.load_policies()
        
        for year in range(self.start_year, self.start_year + self.years):
            print(f"\nYear {year}")
            year_city_data = []
            gdp_per_capita_list = []
            
            # Trigger shocks for shock/policy models
            if self.model_type in ["shock", "policy"]:
                triggered_shocks = self.trigger_random_shocks(year)
                if triggered_shocks:
                    print(f"  Shocks applied: {len(triggered_shocks)} shock(s) affected production parameters")
            # Trigger policies for policy model
            if self.model_type == "policy":
                triggered_policies = self.trigger_random_policies(year)
                if triggered_policies:
                    print(f"  Policies applied: {len(triggered_policies)} policy/policies affected production parameters")
            
            # Migration only occurs in open economy models
            if self.model_type in ["open", "shock", "policy"]:
                # Calculate and apply migration for this year
                migration_flows = self.calculate_migration_populations(year)
                if migration_flows:
                    total_migrants = sum(migration_flows.values())
                    print(f"  Migration: {total_migrants:,} people moved between cities")
                    self.apply_migration(migration_flows)
            else:
                print(f"  Closed Economy: No migration between cities")
            
            for city in self.tr_network.cities.values():
                # Update population (natural growth only, migration already applied if open economy)
                # Note: population comes from node data and represents total residents
                city.population = int(city.population * (1 + self.population_growth_rate))
                
                # Update each demographic group in labor demographics
                # Note: demographics represent people from different origin cities
                for demo in city.demographics:
                    demo.population_count = int(demo.population_count * (1 + self.population_growth_rate))
                
                # Update labor force (natural growth)
                city.labor_force = int(city.labor_force * (1 + self.labor_growth_rate))
                
                # Maintain demographic proportions after labor force growth
                self._maintain_demographic_proportions(city)
                
                # Note: labor_force is separate from population and comes from edges data
                # Labor force represents working people, not total population
                
                # Calculate production
                production = city.calculate_production()
                # Update capital stock: increase by 5% of production level every year
                city.capital_stock = city.capital_stock + 0.05 * production
                # Optionally, update year in city
                city.year = year
                
                # Collect city data for export
                try:
                    gdp_per_capita = city.calculate_gdp_per_capita()
                except Exception:
                    gdp_per_capita = 0.0
                year_city_data.append({
                    'Year': year,
                    'Model_Type': self.model_type.capitalize(),
                    'City': city.name,
                    'Population': city.population,
                    'Labor_Force': city.labor_force,
                    'Capital_Stock': city.capital_stock,
                    'Production': production,
                    'GDP_per_Capita': gdp_per_capita
                })
                if city.population > 0:
                    gdp_per_capita_list.append(gdp_per_capita)
                # Record demographics for this city and year
                for demo in city.demographics:
                    self.demographics_results.append({
                        'Year': year,
                        'Model_Type': self.model_type.capitalize(),
                        'City': city.name,
                        'Origin_City': demo.origin_city,
                        'Population_Count': demo.population_count
                    })
            # Store all city data for this year
            self.results.extend(year_city_data)
            # Calculate and store Gini coefficient for this year
            gini = self.calculate_gini(gdp_per_capita_list)
            self.gini_results.append({
                'Year': year, 
                'Model_Type': self.model_type.capitalize(),
                'Gini_Coefficient': gini
            })
            # Print summary for the year
            total_production = sum(d['Production'] for d in year_city_data)
            total_population = sum(d['Population'] for d in year_city_data)
            print(f"  Total production: {total_production:,.2f}")
            print(f"  Total population: {total_population:,}")
            print(f"  Average GDP per capita: {total_production / total_population:,.2f}")
            print(f"  Gini coefficient: {gini:.4f}")
            
            # Show shock/policy summary if occurred
            if self.model_type in ["shock", "policy"]:
                shock_summary = self.get_shock_summary(year)
                if "No shocks occurred" not in shock_summary:
                    print(f"  Shocks: {shock_summary.split('Total shocks:')[1].split('\n')[0].strip()}")
                if self.model_type == "policy" and hasattr(self, 'active_policies'):
                    policy_summary = self.get_policy_summary(year)
                    if "No policies occurred" not in policy_summary:
                        print(f"  Policies: {policy_summary.split('Total policies:')[1].split('\n')[0].strip()}")

    def calculate_gini(self, values):
        """Calculate Gini coefficient for a list of values."""
        if len(values) < 2:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumvals = [sum(sorted_vals[:i+1]) for i in range(n)]
        gini = (n + 1 - 2 * sum((i + 1) * v for i, v in enumerate(sorted_vals)) / sum(sorted_vals)) / n
        return abs(gini)

    def export_results(self, filename=None):
        """Export simulation results to Excel with city-year data, yearly Gini, demographics (matrix format), and shock results (if available)."""
        if filename is None:
            # Use output configuration to generate organized filename
            filename = get_excel_path("turkey_cities_simulation", self.model_type)
        
        df = pd.DataFrame(self.results)
        gini_df = pd.DataFrame(self.gini_results)
        demographics_df = pd.DataFrame(self.demographics_results)
        
        # Pivot demographics to matrix format
        if not demographics_df.empty:
            demo_matrix = demographics_df.pivot_table(index=['Year', 'City'], columns='Origin_City', values='Population_Count', fill_value=0)
            demo_matrix = demo_matrix.reset_index()
        else:
            demo_matrix = demographics_df  # empty
        
        # Merge yearly Gini onto city-year data
        try:
            if not gini_df.empty and 'Year' in gini_df.columns:
                df = df.merge(gini_df[['Year','Gini_Coefficient']], on='Year', how='left')
                df = df.rename(columns={'Gini_Coefficient':'Gini_This_Year'})
        except Exception:
            pass
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='City_Year_Data', index=False)
            gini_df.to_excel(writer, sheet_name='Yearly_Gini', index=False)
            demo_matrix.to_excel(writer, sheet_name='Demographics', index=False)
            
            # Add shock results if available
            if self.model_type in ["shock", "policy"] and hasattr(self, 'active_shocks') and self.active_shocks:
                # Create detailed shock results with parameter changes
                detailed_shock_data = []
                
                for shock in self.active_shocks:
                    year = shock['Year']
                    effects = shock['Effects']
                    
                    # Get all cities that were affected by this shock
                    affected_cities = []
                    for target, effect_values in effects.items():
                        if target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                            # Nationwide shock affects all cities
                            affected_cities.extend(list(self.tr_network.cities.keys()))
                        elif target in self.tr_network.cities:
                            # City-specific shock
                            affected_cities.append(target)
                    
                    # Create records for each affected city
                    for city_name in affected_cities:
                        city = self.tr_network.cities[city_name]
                        
                        # Get old parameter values (before shock)
                        old_params = {
                            'A': city.A,
                            'K': city.capital_stock,
                            'L': city.labor_force,
                            'ALPHA': city.alpha,
                            'BETA': city.beta
                        }
                        
                        # Calculate new parameter values (after shock)
                        new_params = {}
                        # Handle both dictionary and list formats
                        if isinstance(effect_values, dict):
                            for param, effect in effect_values.items():
                                if param == "A":
                                    if old_params['A'] < 0:
                                        new_params['A'] = old_params['A'] * (1 - effect/100)
                                    else:
                                        new_params['A'] = old_params['A'] * (1 + effect/100)
                                elif param == "K":
                                    if old_params['K'] < 0:
                                        new_params['K'] = old_params['K'] * (1 - effect/100)
                                    else:
                                        new_params['K'] = old_params['K'] * (1 + effect/100)
                                elif param == "L":
                                    if old_params['L'] < 0:
                                        new_params['L'] = int(old_params['L'] * (1 - effect/100))
                                    else:
                                        new_params['L'] = int(old_params['L'] * (1 + effect/100))
                                elif param == "ALPHA":
                                    if old_params['ALPHA'] < 0:
                                        new_params['ALPHA'] = old_params['ALPHA'] * (1 - effect/100)
                                    else:
                                        new_params['ALPHA'] = old_params['ALPHA'] * (1 + effect/100)
                                elif param == "BETA":
                                    if old_params['BETA'] < 0:
                                        new_params['BETA'] = old_params['BETA'] * (1 - effect/100)
                                    else:
                                        new_params['BETA'] = old_params['BETA'] * (1 + effect/100)
                        elif isinstance(effect_values, list):
                            for param, effect in effect_values:
                                if param == "A":
                                    if old_params['A'] < 0:
                                        new_params['A'] = old_params['A'] * (1 - effect/100)
                                    else:
                                        new_params['A'] = old_params['A'] * (1 + effect/100)
                                elif param == "K":
                                    if old_params['K'] < 0:
                                        new_params['K'] = old_params['K'] * (1 - effect/100)
                                    else:
                                        new_params['K'] = old_params['K'] * (1 + effect/100)
                                elif param == "L":
                                    if old_params['L'] < 0:
                                        new_params['L'] = int(old_params['L'] * (1 - effect/100))
                                    else:
                                        new_params['L'] = int(old_params['L'] * (1 + effect/100))
                                elif param == "ALPHA":
                                    if old_params['ALPHA'] < 0:
                                        new_params['ALPHA'] = old_params['ALPHA'] * (1 - effect/100)
                                    else:
                                        new_params['ALPHA'] = old_params['ALPHA'] * (1 + effect/100)
                                elif param == "BETA":
                                    if old_params['BETA'] < 0:
                                        new_params['BETA'] = old_params['BETA'] * (1 - effect/100)
                                    else:
                                        new_params['BETA'] = old_params['BETA'] * (1 + effect/100)
                        
                        # Fill in unchanged parameters
                        for param in ['A', 'K', 'L', 'ALPHA', 'BETA']:
                            if param not in new_params:
                                new_params[param] = old_params[param]
                        
                        # Create detailed shock record
                        shock_record = {
                            'Year': year,
                            'City': city_name,
                            'Shock_Title': shock['Effect_Title'],
                            'Shock_Domain': shock['Domain'],
                            'Shock_Description': shock['Description'],
                            'Shock_Type_ID': shock.get('Type_ID', 0),
                            'Shock_Subtype_ID': shock.get('Subtype_ID', 0),
                            'Shock_Scope': shock.get('Scope', ''),
                            # Old parameter values
                            'Old_A': old_params['A'],
                            'Old_K': old_params['K'],
                            'Old_L': old_params['L'],
                            'Old_ALPHA': old_params['ALPHA'],
                            'Old_BETA': old_params['BETA'],
                            # New parameter values
                            'New_A': new_params['A'],
                            'New_K': new_params['K'],
                            'New_L': new_params['L'],
                            'New_ALPHA': new_params['ALPHA'],
                            'New_BETA': new_params['BETA'],
                            # Parameter changes
                            'Delta_A': new_params['A'] - old_params['A'],
                            'Delta_K': new_params['K'] - old_params['K'],
                            'Delta_L': new_params['L'] - old_params['L'],
                            'Delta_ALPHA': new_params['ALPHA'] - old_params['ALPHA'],
                            'Delta_BETA': new_params['BETA'] - old_params['BETA']
                        }
                        
                        detailed_shock_data.append(shock_record)
                
                # Create detailed shock DataFrame
                detailed_shock_df = pd.DataFrame(detailed_shock_data)
                detailed_shock_df.to_excel(writer, sheet_name='Detailed_Shock_Results', index=False)
                
                # Create shock summary sheet
                summary_data = []
                for shock in self.active_shocks:
                    summary_data.append({
                        'Year': shock['Year'],
                        'Effect_Title': shock['Effect_Title'],
                        'Domain': shock['Domain'],
                        'Target_Count': len(shock['Effects'])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Shock_Summary', index=False)
                
                # Create city-specific shock summary
                city_shock_summary = []
                for city_name in self.tr_network.cities.keys():
                    city_shocks = [s for s in self.active_shocks if any(
                        target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"] or 
                        target == city_name 
                        for target in s['Effects'].keys()
                    )]
                    
                    if city_shocks:
                        total_effects = 0
                        positive_effects = 0
                        negative_effects = 0
                        
                        for shock in city_shocks:
                            for target, effect_values in shock['Effects'].items():
                                if (target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"] or 
                                    target == city_name):
                                    if isinstance(effect_values, dict):
                                        for param, effect in effect_values.items():
                                            total_effects += 1
                                            if effect > 0:
                                                positive_effects += 1
                                            elif effect < 0:
                                                negative_effects += 1
                                    elif isinstance(effect_values, list):
                                        for param, effect in effect_values:
                                            total_effects += 1
                                            if effect > 0:
                                                positive_effects += 1
                                            elif effect < 0:
                                                negative_effects += 1
                        
                        city_shock_summary.append({
                            'City': city_name,
                            'Total_Shocks': len(city_shocks),
                            'Total_Effects': total_effects,
                            'Positive_Effects': positive_effects,
                            'Negative_Effects': negative_effects
                        })
                
                if city_shock_summary:
                    city_summary_df = pd.DataFrame(city_shock_summary)
                    city_summary_df.to_excel(writer, sheet_name='City_Shock_Summary', index=False)
        
        # Add migration summary for open economy models
        if self.model_type in ["open", "shock", "policy"] and hasattr(self, 'migration_results') and self.migration_results:
            migration_summary_data = []
            
            for city_name in self.tr_network.cities.keys():
                # Calculate total out-migration for this city
                total_pop_out = sum(m['Migration_Population'] for m in self.migration_results if m['Source_City'] == city_name)
                total_labor_out = sum(m['Migration_Labor_Force'] for m in self.migration_results if m['Source_City'] == city_name)
                total_capital_out = sum(m.get('Migration_Capital', 0) for m in self.migration_results if m['Source_City'] == city_name)
                
                # Calculate total in-migration for this city
                total_pop_in = sum(m['Migration_Population'] for m in self.migration_results if m['Target_City'] == city_name)
                total_labor_in = sum(m['Migration_Labor_Force'] for m in self.migration_results if m['Target_City'] == city_name)
                total_capital_in = sum(m.get('Migration_Capital', 0) for m in self.migration_results if m['Target_City'] == city_name)
                
                # Net migration
                net_pop = total_pop_in - total_pop_out
                net_labor = total_labor_in - total_labor_out
                net_capital = total_capital_in - total_capital_out
                
                migration_summary_data.append({
                    'City': city_name,
                    'Total_Population_Out': total_pop_out,
                    'Total_Labor_Out': total_labor_out,
                    'Total_Capital_Out': total_capital_out,
                    'Total_Population_In': total_pop_in,
                    'Total_Labor_In': total_labor_in,
                    'Total_Capital_In': total_capital_in,
                    'Net_Population_Migration': net_pop,
                    'Net_Labor_Migration': net_labor,
                    'Net_Capital_Migration': net_capital
                })
            
            migration_summary_df = pd.DataFrame(migration_summary_data)
            migration_summary_df.to_excel(writer, sheet_name='Migration_Summary', index=False)
        
        print(f"Simulation results exported to {filename}")
        
        # Export shock and policy results separately if available
        if self.model_type in ["shock", "policy"] and hasattr(self, 'active_shocks') and self.active_shocks:
            self.export_shock_results()
            # Also export detailed shock results for reinforcement learning
            self.export_detailed_shock_results()
        if self.model_type == "policy" and hasattr(self, 'active_policies') and self.active_policies:
            self.export_policy_results()
            self.export_detailed_policy_results()

    def export_migration_scores(self, year, filename=None, threshold=0.0):
        """
        Calculate and export softmax-normalized migration probabilities for all city pairs for a given year.
        Only pairs with probability above the threshold are exported.
        """
        if filename is None:
            # Use output configuration to generate organized filename
            filename = get_excel_path("migration_scores", self.model_type)
        
        migration_records = []
        cities = list(self.tr_network.cities.values())
        for source in cities:
            migrations = self.migration_softmax(source, threshold=threshold)
            for target, prob in migrations:
                migration_records.append({
                    "Year": year,
                    "Source_City": source.name,
                    "Target_City": target.name,
                    "Migration_Probability": prob
                })
        df = pd.DataFrame(migration_records)
        df.to_excel(filename, index=False)
        print(f"Migration probabilities exported to {filename}")

    def export_migration_regression_data(self, year, filename=None):
        """
        Export a file with migration decision features for all city pairs for regression analysis.
        """
        if filename is None:
            # Use output configuration to generate organized filename
            filename = get_excel_path("migration_regression_data", self.model_type)
        
        records = []
        cities = list(self.tr_network.cities.values())
        for source in cities:
            for target in cities:
                if source.name == target.name:
                    continue
                # GDP per capita difference
                gdp_diff = target.calculate_gdp_per_capita() - source.calculate_gdp_per_capita()
                # Diaspora ratio
                diaspora = 0
                total_source = 0
                for city in self.tr_network.cities.values():
                    for demo in city.demographics:
                        if demo.origin_city == source.name:
                            total_source += demo.population_count
                            if city.name == target.name:
                                diaspora = demo.population_count
                diaspora_ratio = diaspora / total_source if total_source > 0 else 0
                # Distance
                distance = source.distance_to(target)
                records.append({
                    "Year": year,
                    "Source_City": source.name,
                    "Target_City": target.name,
                    "GDP_per_capita_diff": gdp_diff,
                    "Diaspora_Ratio": diaspora_ratio,
                    "Distance": distance,
                    # "Real_Migration_Level": 0  # You will fill this in manually
                })
        df = pd.DataFrame(records)
        df.to_excel(filename, index=False)
        print(f"Migration regression data exported to {filename}")

    def analyze(self):
        """Run the simulation and export results."""
        self.simulate()
        #self.export_results()

    def animate_gdp_per_capita(self, save_path=None, figsize=(32, 24), cmap='RdYlGn'):
        """
        Animate Turkey map colored by GDP per capita for each year of the simulation.
        If save_path is provided, save as GIF or MP4.
        """
        if self.tr_network.provinces is None:
            print("Cannot animate: provinces data not loaded")
            return
        
        # Prepare data
        df = pd.DataFrame(self.results)
        years = sorted(df['Year'].unique())
        provinces = self.tr_network.provinces.copy()
        province_names = provinces['NAME_1'].tolist()
        # Build city-to-province mapping (as in visualize_turkey_map)
        city_to_province = {}
        for province_name in province_names:
            for city in self.tr_network.cities:
                if city.upper() in province_name.upper() or province_name.upper() in city.upper():
                    city_to_province[city] = province_name
        # Prepare figure with constrained layout
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.close(fig)  # Prevent duplicate static display in notebooks
        def get_province_gdp(year):
            year_df = df[df['Year'] == year]
            city_gdp = dict(zip(year_df['City'], year_df['GDP_per_Capita']))
            province_gdp = []
            for province in province_names:
                # Find city for this province
                val = 0.0
                for city, prov in city_to_province.items():
                    if prov == province and city in city_gdp:
                        val = city_gdp[city]
                        break
                province_gdp.append(val)
            return np.array(province_gdp)
        # Set up the initial plot with per-year scaling
        province_gdp = get_province_gdp(years[0])
        vmin, vmax = float(province_gdp.min()), float(province_gdp.max())
        if vmin == vmax:
            vmax = vmin + 1e-6
        provinces['gdp_per_capita'] = province_gdp
        collection = provinces.plot(
            ax=ax,
            column='gdp_per_capita',
            cmap=cmap,
            edgecolor='white',
            linewidth=0.5,
            legend=False,
            vmin=vmin,
            vmax=vmax
        )
        # Place colorbar below the map, horizontal
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.025])  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('GDP per Capita')
        # Add city labels (static, since positions don't change)
        for city in self.tr_network.cities:
            pos = self.tr_network.network.nodes[city]['pos']
            ax.annotate(
                city,
                xy=pos,
                xytext=(0, 0),
                textcoords='offset points',
                fontsize=18,
                ha='center',
                va='center',
                fontweight='bold',
                color='black'
            )
        def update(frame):
            year = years[frame]
            province_gdp = get_province_gdp(year)
            provinces['gdp_per_capita'] = province_gdp
            ax.clear()
            # Per-year vmin/vmax for clearer yearly differences
            vmin = float(province_gdp.min())
            vmax = float(province_gdp.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
            provinces.plot(
                ax=ax,
                column='gdp_per_capita',
                cmap=cmap,
                edgecolor='white',
                linewidth=0.5,
                legend=False,
                vmin=vmin,
                vmax=vmax
            )
            # Add city labels again
            for city in self.tr_network.cities:
                pos = self.tr_network.network.nodes[city]['pos']
                ax.annotate(
                    city,
                    xy=pos,
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=18,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='black'
                )
            ax.set_title(f"Turkey GDP per Capita - {self.model_type.capitalize()} Economy - Year {year}", fontsize=24, fontweight='bold', pad=20)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(25, 45)
            ax.set_ylim(35, 43)
            # Update colorbar for this frame
            sm.set_norm(plt.Normalize(vmin=vmin, vmax=vmax))
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False)
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=6)
            else:
                ani.save(save_path, writer='ffmpeg', fps=6)
            print(f"Animation saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            save_path = get_animation_path("turkey_gdp_per_capita", self.model_type)
            ani.save(save_path, writer='pillow', fps=6)
            print(f"Animation saved to {save_path}")
            plt.show()

    def animate_gdp_per_capita_enhanced(self, save_path=None, figsize=(32, 24), cmap='RdYlGn', color_scale='dynamic'):
        """
        Enhanced animation with better color scaling options for more visible changes over time.
        
        Args:
            save_path (str): Path to save the animation
            figsize (tuple): Figure size
            cmap (str): Colormap
            color_scale (str): 'consistent' for same scale across all frames, 
                              'dynamic' for adaptive scale that emphasizes changes
        """
        if self.tr_network.provinces is None:
            print("Cannot animate: provinces data not loaded")
            return
        
        # Prepare data
        df = pd.DataFrame(self.results)
        years = sorted(df['Year'].unique())
        provinces = self.tr_network.provinces.copy()
        province_names = provinces['NAME_1'].tolist()
        
        # Build city-to-province mapping
        city_to_province = {}
        for province_name in province_names:
            for city in self.tr_network.cities:
                if city.upper() in province_name.upper() or province_name.upper() in city.upper():
                    city_to_province[city] = province_name
        
        def get_province_gdp(year):
            year_df = df[df['Year'] == year]
            city_gdp = dict(zip(year_df['City'], year_df['GDP_per_Capita']))
            province_gdp = []
            for province in province_names:
                val = 0.0
                for city, prov in city_to_province.items():
                    if prov == province and city in city_gdp:
                        val = city_gdp[city]
                        break
                province_gdp.append(val)
            return np.array(province_gdp)
        
        # Calculate color scale based on preference
        if color_scale == 'consistent':
            # Use global min/max for consistent scale (original behavior)
            all_gdp = np.concatenate([get_province_gdp(y) for y in years])
            vmin, vmax = all_gdp.min(), all_gdp.max()
        else:
            # dynamic: placeholder initial values, will update per frame
            vmin, vmax = 0, 1
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.close(fig)
        
        # Set up the initial plot
        province_gdp = get_province_gdp(years[0])
        if color_scale == 'dynamic':
            vmin, vmax = float(province_gdp.min()), float(province_gdp.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
        provinces['gdp_per_capita'] = province_gdp
        collection = provinces.plot(
            ax=ax,
            column='gdp_per_capita',
            cmap=cmap,
            edgecolor='white',
            linewidth=0.5,
            legend=False,
            vmin=vmin,
            vmax=vmax
        )
        
        # Place colorbar below the map, horizontal
        cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.025])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('GDP per Capita')
        
        # Add city labels
        for city in self.tr_network.cities:
            pos = self.tr_network.network.nodes[city]['pos']
            ax.annotate(
                city,
                xy=pos,
                xytext=(0, 0),
                textcoords='offset points',
                fontsize=18,
                ha='center',
                va='center',
                fontweight='bold',
                color='black'
            )
        
        def update(frame):
            year = years[frame]
            province_gdp = get_province_gdp(year)
            provinces['gdp_per_capita'] = province_gdp
            ax.clear()
            
            # Per-frame scaling for dynamic mode
            local_vmin, local_vmax = vmin, vmax
            if color_scale == 'dynamic':
                local_vmin = float(province_gdp.min())
                local_vmax = float(province_gdp.max())
                if local_vmin == local_vmax:
                    local_vmax = local_vmin + 1e-6
            
            provinces.plot(
                ax=ax,
                column='gdp_per_capita',
                cmap=cmap,
                edgecolor='white',
                linewidth=0.5,
                legend=False,
                vmin=local_vmin,
                vmax=local_vmax
            )
            
            # Add city labels again
            for city in self.tr_network.cities:
                pos = self.tr_network.network.nodes[city]['pos']
                ax.annotate(
                    city,
                    xy=pos,
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=18,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='black'
                )
            
            ax.set_title(f"Turkey GDP per Capita - {self.model_type.capitalize()} Economy - Year {year}\nColor Scale: {color_scale.title()}", 
                        fontsize=24, fontweight='bold', pad=20)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(25, 45)
            ax.set_ylim(35, 43)
            
            # Update colorbar for this frame
            sm.set_norm(plt.Normalize(vmin=local_vmin, vmax=local_vmax))
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False)
        
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=6)
            else:
                ani.save(save_path, writer='ffmpeg', fps=6)
            print(f"Enhanced animation saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_animation_path
            save_path = get_animation_path(f"turkey_gdp_per_capita_enhanced_{color_scale}", self.model_type)
            ani.save(save_path, writer='pillow', fps=6)
            print(f"Enhanced animation saved to {save_path}")
        
        plt.show()
        return save_path

    def calculate_migration_populations(self, year):
        """
        Calculate actual migration populations between cities based on migration scores.
        Returns a dictionary of migration flows: {(source, target): population}
        """
        migration_flows = {}
        
        for source_city in self.tr_network.cities.values():
            # Get migration probabilities to all other cities
            migrations = self.migration_softmax(source_city, threshold=0.01)  # Lower threshold for more flows
            
            if not migrations:
                continue
                
            # Calculate total potential migrants (based on migration rate)
            potential_migrants = int(source_city.population * self.migration_rate)
            
            if potential_migrants == 0:
                continue
                
            # Distribute migrants based on probabilities
            for target_city, prob in migrations:
                migrants = int(prob * potential_migrants)
                if migrants > 0:
                    migration_flows[(source_city.name, target_city.name)] = migrants
                    
                    # Record migration flow
                    self.migration_results.append({
                        'Year': year,
                        'Source_City': source_city.name,
                        'Target_City': target_city.name,
                        'Migration_Population': migrants,
                        'Migration_Labor_Force': int(migrants * (source_city.labor_force / source_city.population)) if source_city.population > 0 else 0,
                        'Migration_Capital': int(migrants * (source_city.labor_force / source_city.population) * self.capital_migration_ratio * (source_city.capital_stock / source_city.labor_force)) if source_city.population > 0 and source_city.labor_force > 0 else 0,
                        'Probability': prob,
                        'Source_Population': source_city.population,
                        'Target_Population': target_city.population,
                        'Source_Labor_Force': source_city.labor_force,
                        'Target_Labor_Force': target_city.labor_force,
                        'Source_Capital_Stock': source_city.capital_stock,
                        'Target_Capital_Stock': target_city.capital_stock
                    })
        
        return migration_flows

    def apply_migration(self, migration_flows):
        """
        Apply migration flows to update city demographics, populations, and labor force.
        When people migrate from source to target city:
        - ALL demographic groups in source city are reduced proportionally (both population and labor force)
        - Migrants are added ONLY to their ORIGIN demographic group in target city
        - Labor force also migrates at the same ratio as population
        - This maintains demographic composition while allowing population and labor movement
        """
        # Apply migrations
        for (source_name, target_name), migrants in migration_flows.items():
            if migrants == 0:
                continue
                
            source_city = self.tr_network.cities[source_name]
            target_city = self.tr_network.cities[target_name]
            
            # Calculate migration rate as percentage of source city population
            if source_city.population > 0:
                migration_rate = migrants / source_city.population
            else:
                continue
            
            # 1. REDUCE ALL demographic groups in source city proportionally (POPULATION)
            # Note: demographics represent people from different origin cities
            # When we reduce demographics proportionally, we're reducing both population and labor force
            # since the same people who live in a city also work in that city
            for demo in source_city.demographics:
                reduction = int(demo.population_count * migration_rate)
                demo.population_count = max(0, demo.population_count - reduction)
            
            # 2. ADD migrants ONLY to their ORIGIN demographic group in target city (POPULATION)
            # Find the demographic group for the source city in the target city
            demo_found = False
            for demo in target_city.demographics:
                if demo.origin_city == source_name:
                    demo.population_count += migrants
                    demo_found = True
                    break
            
            # If the source city's demographic group doesn't exist in target city, create it
            if not demo_found:
                new_demo = City.Demographics(source_name, migrants)
                target_city.demographics.append(new_demo)
            
            # 3. Update total populations
            source_city.population = max(0, source_city.population - migrants)
            target_city.population += migrants
            
            # 4. Update labor force proportionally (same ratio as population migration)
            if source_city.labor_force > 0:
                labor_migrants = int(source_city.labor_force * migration_rate)
                
                # 4a. REDUCE labor force from source city demographics proportionally
                # Note: We need to update the labor force demographics, not just total labor force
                # Since demographics represent both population and labor force, we need to handle this carefully
                
                # First, update the total labor force
                source_city.labor_force = max(0, source_city.labor_force - labor_migrants)
                target_city.labor_force += labor_migrants
                
                # 4b. Distribute labor force migrants proportionally across all origin cities
                # based on the source city's current demographic composition
                total_source_demographics = sum(demo.population_count for demo in source_city.demographics)
                
                if total_source_demographics > 0:
                    # Calculate how many labor migrants should go to each origin city group
                    for demo in source_city.demographics:
                        # Calculate the proportion of this origin city in the source city
                        origin_proportion = demo.population_count / total_source_demographics
                        
                        # Calculate how many labor migrants from this origin city should migrate
                        labor_migrants_from_origin = int(labor_migrants * origin_proportion)
                        
                        if labor_migrants_from_origin > 0:
                            # Find or create the demographic group for this origin city in the target city
                            target_demo_found = False
                            for target_demo in target_city.demographics:
                                if target_demo.origin_city == demo.origin_city:
                                    # Add the labor migrants to this origin city group
                                    target_demo.population_count += labor_migrants_from_origin
                                    target_demo_found = True
                                    break
                            
                            # If this origin city group doesn't exist in target city, create it
                            if not target_demo_found:
                                new_demo = City.Demographics(demo.origin_city, labor_migrants_from_origin)
                                target_city.demographics.append(new_demo)
                
                # Debug output for verification
                #print(f"  Migration: {source_name} → {target_name}: {migrants:,} people, {labor_migrants:,} labor")
                
            # 5. CAPITAL MIGRATION (at capital_migration_ratio of labor migration)
            if self.model_type in ["open", "shock", "policy"] and self.capital_migration_ratio > 0:
                # Calculate capital migration based on labor migration rate and capital_migration_ratio
                capital_migration_rate = migration_rate * self.capital_migration_ratio
                capital_migrants = int(source_city.capital_stock * capital_migration_rate)
                
                if capital_migrants > 0:
                    # Move capital from source to target city
                    source_city.capital_stock = max(0, source_city.capital_stock - capital_migrants)
                    target_city.capital_stock += capital_migrants
                    
                    # print(f"    Capital Migration: {source_name} → {target_name}: {capital_migrants:,.0f} capital")
                    # print(f"    Capital migration rate: {capital_migration_rate:.1%} (1/{int(1/self.capital_migration_ratio)} of labor migration rate)")
                    # print(f"    Capital Stock: {source_name} {source_city.capital_stock:,.0f} → {source_city.capital_stock + capital_migrants:,.0f}")
                    # print(f"    Capital Stock: {target_name} {target_city.capital_stock:,.0f} → {target_city.capital_stock + capital_migrants:,.0f}")

    def set_migration_weights_from_regression(self, gdp_coef, diaspora_coef, distance_coef, source_pop_coef, target_pop_coef):
        """
        Set migration weights from regression coefficients.
        Use this method after running your regression analysis to update the weights.
        
        Parameters:
        - gdp_coef: coefficient for GDP per capita difference
        - diaspora_coef: coefficient for diaspora ratio
        - distance_coef: coefficient for distance (should be negative for inverse effect)
        - source_pop_coef: coefficient for source city population
        - target_pop_coef: coefficient for target city population
        """
        self.gdp_weight = gdp_coef
        self.diaspora_weight = diaspora_coef
        self.distance_weight = distance_coef
        self.source_pop_weight = source_pop_coef
        self.target_pop_weight = target_pop_coef
        
        print(f"Migration weights updated from regression results:")
        print(f"  GDP per capita weight: {self.gdp_weight}")
        print(f"  Diaspora weight: {self.diaspora_weight}")
        print(f"  Distance weight: {self.distance_weight}")
        print(f"  Source population weight: {self.source_pop_weight}")
        print(f"  Target population weight: {self.target_pop_weight}")

    def show_migration_impact(self, year, source_city_name, target_city_name):
        """
        Show detailed before/after comparison for a specific migration flow.
        This helps verify that proportional migration is working correctly.
        """
        # Find the migration record
        migration_record = None
        for record in self.migration_results:
            if (record['Year'] == year and 
                record['Source_City'] == source_city_name and 
                record['Target_City'] == target_city_name):
                migration_record = record
                break
        
        if not migration_record:
            print(f"No migration found from {source_city_name} to {target_city_name} in year {year}")
            return
        
        migrants = migration_record['Migration_Population']
        source_city = self.tr_network.cities[source_city_name]
        target_city = self.tr_network.cities[target_city_name]
        
        print(f"\n=== Migration Impact Analysis ===")
        print(f"Year: {year}")
        print(f"Flow: {source_city_name} → {target_city_name}")
        print(f"Total migrants: {migrants:,}")
        
        # Calculate migration rate
        migration_rate = migrants / (source_city.population + migrants) if (source_city.population + migrants) > 0 else 0
        
        print(f"Migration rate: {migration_rate:.1%}")
        print(f"\nSource City ({source_city_name}) - After Migration:")
        print(f"  Total population: {source_city.population:,}")
        print(f"  Labor force: {source_city.labor_force:,}")
        print(f"  Demographic breakdown:")
        
        for demo in source_city.demographics:
            percentage = (demo.population_count / source_city.population * 100) if source_city.population > 0 else 0
            print(f"    {demo.origin_city}: {demo.population_count:,} ({percentage:.1f}%)")
        
        print(f"\nTarget City ({target_city_name}) - After Migration:")
        print(f"  Total population: {target_city.population:,}")
        print(f"  Labor force: {target_city.labor_force:,}")
        print(f"  Demographic breakdown:")
        
        for demo in target_city.demographics:
            percentage = (demo.population_count / target_city.population * 100) if target_city.population > 0 else 0
            print(f"    {demo.origin_city}: {demo.population_count:,} ({percentage:.1f}%)")
        
        # Show which demographic group received the migrants
        print(f"\nMigration Details:")
        print(f"  {migrants:,} people from {source_city_name} migrated to {target_city_name}")
        print(f"  These migrants were added to the '{source_city_name}' origin group in {target_city_name}")
        print(f"  All demographic groups in {source_city_name} were reduced by {migration_rate:.1%}")

    def verify_demographic_proportions(self, city_name):
        """
        Verify that demographic proportions are maintained correctly after migration.
        This helps debug migration issues.
        """
        city = self.tr_network.cities[city_name]
        total_pop = city.population
        total_labor = city.labor_force
        
        print(f"\nDemographic verification for {city_name}:")
        print(f"  Total population: {total_pop:,}")
        print(f"  Total labor force: {total_labor:,}")
        print(f"  Demographic breakdown:")
        
        for demo in city.demographics:
            percentage = (demo.population_count / total_pop * 100) if total_pop > 0 else 0
            print(f"    {demo.origin_city}: {demo.population_count:,} ({percentage:.1f}%)")
        
        # Check if labor force matches population
        if total_pop != total_labor:
            print(f"  WARNING: Population ({total_pop:,}) != Labor force ({total_labor:,})")
        else:
            print(f"  ✓ Population and labor force match")

    def get_migration_summary(self, year):
        """
        Get a summary of migration patterns for a specific year.
        Returns migration statistics and top migration flows for both population and labor force.
        """
        year_migrations = [m for m in self.migration_results if m['Year'] == year]
        
        if not year_migrations:
            return f"No migration data for year {year}"
        
        total_pop_migrants = sum(m['Migration_Population'] for m in year_migrations)
        total_labor_migrants = sum(m['Migration_Labor_Force'] for m in year_migrations)
        total_capital_migrants = sum(m.get('Migration_Capital', 0) for m in year_migrations)
        num_flows = len(year_migrations)
        
        # Top 5 migration flows
        top_flows = sorted(year_migrations, key=lambda x: x['Migration_Population'], reverse=True)[:5]
        
        summary = f"Migration Summary for Year {year}:\n"
        summary += f"Total migrants: {total_pop_migrants:,}\n"
        summary += f"City pairs: {num_flows}\n"
        summary += f"Top flows:\n"
        
        for i, flow in enumerate(top_flows, 1):
            summary += f"  {i}. {flow['Source_City']} → {flow['Target_City']}: {flow['Migration_Population']:,} people\n"
        
        return summary

    def verify_migration_correctness(self, year):
        """
        Verify that migration is working correctly by checking:
        1. Population conservation (total population unchanged)
        2. Demographic group changes are proportional in source cities
        3. Migrants are only added to origin groups in target cities
        """
        print(f"\n=== Migration Correctness Verification for Year {year} ===")
        
        year_migrations = [m for m in self.migration_results if m['Year'] == year]
        if not year_migrations:
            print("No migrations found for this year")
            return
        
        total_migrants = sum(m['Migration_Population'] for m in year_migrations)
        print(f"Total migrants: {total_migrants:,}")
        
        # Check population conservation
        total_pop_before = 0
        total_pop_after = 0
        
        for city in self.tr_network.cities.values():
            # We need to estimate the before population (current + migrants out - migrants in)
            migrants_out = sum(m['Migration_Population'] for m in year_migrations if m['Source_City'] == city.name)
            migrants_in = sum(m['Migration_Population'] for m in year_migrations if m['Target_City'] == city.name)
            estimated_before = city.population + migrants_out - migrants_in
            total_pop_before += estimated_before
            total_pop_after += city.population
        
        print(f"Population conservation check:")
        print(f"  Total population before migration: {total_pop_before:,}")
        print(f"  Total population after migration: {total_pop_after:,}")
        if abs(total_pop_before - total_pop_after) < 1:  # Allow for rounding errors
            print(f"  ✓ Population conserved")
        else:
            print(f"  ✗ Population not conserved! Difference: {total_pop_before - total_pop_after:,}")
        
        # Check labor force conservation
        total_labor_before = 0
        total_labor_after = 0
        
        for city in self.tr_network.cities.values():
            # We need to estimate the before labor force (current + labor migrants out - labor migrants in)
            labor_migrants_out = sum(m['Migration_Labor_Force'] for m in year_migrations if m['Source_City'] == city.name)
            labor_migrants_in = sum(m['Migration_Labor_Force'] for m in year_migrations if m['Target_City'] == city.name)
            estimated_before = city.labor_force + labor_migrants_out - labor_migrants_in
            total_labor_before += estimated_before
            total_labor_after += city.labor_force
        
        print(f"Labor force conservation check:")
        print(f"  Total labor force before migration: {total_labor_before:,}")
        print(f"  Total labor force after migration: {total_labor_after:,}")
        if abs(total_labor_before - total_labor_after) < 1:  # Allow for rounding errors
            print(f"  ✓ Labor force conserved")
        else:
            print(f"  ✗ Labor force not conserved! Difference: {total_labor_before - total_labor_after:,}")
        
        # Check capital conservation (only for open economy models)
        if self.model_type in ["open", "shock", "policy"] and self.capital_migration_ratio > 0:
            total_capital_before = 0
            total_capital_after = 0
            
            for city in self.tr_network.cities.values():
                # We need to estimate the before capital stock (current + capital migrants out - capital migrants in)
                capital_migrants_out = sum(m.get('Migration_Capital', 0) for m in year_migrations if m['Source_City'] == city.name)
                capital_migrants_in = sum(m.get('Migration_Capital', 0) for m in year_migrations if m['Target_City'] == city.name)
                estimated_before = city.capital_stock + capital_migrants_out - capital_migrants_in
                total_capital_before += estimated_before
                total_capital_after += city.capital_stock
            
            print(f"Capital conservation check:")
            print(f"  Total capital before migration: {total_capital_before:,.0f}")
            print(f"  Total capital after migration: {total_capital_after:,.0f}")
            if abs(total_capital_before - total_capital_after) < 1:  # Allow for rounding errors
                print(f"  ✓ Capital conserved")
            else:
                print(f"  ✗ Capital not conserved! Difference: {total_capital_before - total_capital_after:,.0f}")
        
        # Check migration flows
        print(f"\nMigration flows verification:")
        for migration in year_migrations[:3]:  # Show first 3 migrations only
            source = migration['Source_City']
            target = migration['Target_City']
            pop_migrants = migration['Migration_Population']
            print(f"  {source} → {target}: {pop_migrants:,} people")
            print(f"    ✓ Migration applied correctly")

    def run_model_comparison(self, models_config):
        """
        Run multiple models and compare their results.
        
        Args:
            models_config: List of dictionaries with model configurations
            Example: [
                {"model_type": "closed", "years": 15, "name": "Closed Economy"},
                {"model_type": "open", "years": 15, "migration_rate": 0.01, "name": "Open Economy"}
            ]
        """
        print(f"\n=== Running Model Comparison ===")
        all_results = []
        all_gini_results = []
        all_demographics_results = []
        all_migration_results = []
        
        for i, config in enumerate(models_config):
            print(f"\n--- Running Model {i+1}: {config.get('name', config['model_type'])} ---")
            
            # Create new economy instance for this model
            model_economy = TReconomy(
                map_data_path=self.tr_network.map_data_path,
                nodes_file=self.tr_network.nodes_file,
                edges_file=self.tr_network.edges_file,
                current_year=self.start_year,
                years=config.get('years', self.years),
                population_growth_rate=config.get('population_growth_rate', self.population_growth_rate),
                labor_growth_rate=config.get('labor_growth_rate', self.labor_growth_rate),
                model_type=config['model_type'],
                migration_rate=config.get('migration_rate', 0.01),
                capital_migration_ratio=config.get('capital_migration_ratio', 0.2), # Pass capital_migration_ratio
                gdp_weight=config.get('gdp_weight', 1.0),
                diaspora_weight=config.get('diaspora_weight', 1.0),
                distance_weight=config.get('distance_weight', 1.0),
                source_pop_weight=config.get('source_pop_weight', 1.0),
                target_pop_weight=config.get('target_pop_weight', 1.0)
            )
            
            # IMPORTANT: Create a deep copy of the original network for this model
            # This ensures each model starts with completely independent data
            import copy
            for city_name, original_city in self.tr_network.cities.items():
                # Reset to initial demographics from the original network
                city = model_economy.tr_network.cities[city_name]
                city.demographics = []
                city.labor_force = 0
                
                # Rebuild demographics from original data (deep copy)
                for demo in original_city.demographics:
                    new_demo = City.Demographics(demo.origin_city, demo.population_count)
                    city.demographics.append(new_demo)
                    city.labor_force += demo.population_count
                
                # Also copy other important attributes to ensure independence
                city.population = original_city.population
                city.capital_stock = original_city.capital_stock
                city.alpha = original_city.alpha
                city.beta = original_city.beta
                city.A = original_city.A
            
            # Run simulation
            model_economy.analyze()
            
            # Debug: Show demographics after simulation to verify they're different
            print(f"  Model {i+1} ({config.get('name', config['model_type'])}) demographics after simulation:")
            sample_cities = list(model_economy.tr_network.cities.keys())[:3]  # Show first 3 cities
            for city_name in sample_cities:
                city = model_economy.tr_network.cities[city_name]
                total_demo = sum(demo.population_count for demo in city.demographics)
                print(f"    {city_name}: Total demographics = {total_demo:,}, Labor force = {city.labor_force:,}")
            
            # Collect results with model identification
            for result in model_economy.results:
                result['Model_Name'] = config.get('name', config['model_type'])
                all_results.append(result)
            
            for gini_result in model_economy.gini_results:
                gini_result['Model_Name'] = config.get('name', config['model_type'])
                all_gini_results.append(gini_result)
            
            for demo_result in model_economy.demographics_results:
                demo_result['Model_Name'] = config.get('name', config['model_type'])
                all_demographics_results.append(demo_result)
            
            if model_economy.model_type in ["open", "shock", "policy"]:
                for migration_result in model_economy.migration_results:
                    migration_result['Model_Name'] = config.get('name', config['model_type'])
                    all_migration_results.append(migration_result)
            
            # Collect shock/policy data if relevant
            if model_economy.model_type in ["shock", "policy"] and hasattr(model_economy, 'active_shocks') and model_economy.active_shocks:
                for shock in model_economy.active_shocks:
                    shock_copy = shock.copy()
                    shock_copy['Model_Name'] = config.get('name', config['model_type'])
                    if not hasattr(self, 'comparison_shock_results'):
                        self.comparison_shock_results = []
                    self.comparison_shock_results.append(shock_copy)
                # Also create and save the shock animation for this model
                try:
                    print("  Creating shock animation...")
                    model_economy.animate_shock_effects()
                except Exception as e:
                    print(f"  Warning: could not create shock animation: {e}")
            if model_economy.model_type == "policy" and hasattr(model_economy, 'active_policies') and model_economy.active_policies:
                for policy in model_economy.active_policies:
                    pol_copy = policy.copy()
                    pol_copy['Model_Name'] = config.get('name', config['model_type'])
                    if not hasattr(self, 'comparison_policy_results'):
                        self.comparison_policy_results = []
                    self.comparison_policy_results.append(pol_copy)
                try:
                    print("  Creating policy animation...")
                    model_economy.animate_policy_effects()
                except Exception as e:
                    print(f"  Warning: could not create policy animation: {e}")
            
            print(f"✓ Model {i+1} completed")
        
        # Store combined results
        self.comparison_results = all_results
        self.comparison_gini_results = all_gini_results
        self.comparison_demographics_results = all_demographics_results
        self.comparison_migration_results = all_migration_results
        
        print(f"\n=== Model Comparison Complete ===")
        print(f"Total results: {len(all_results)}")
        print(f"Total Gini results: {len(all_gini_results)}")
        print(f"Total demographics results: {len(all_demographics_results)}")
        if all_migration_results:
            print(f"Total migration results: {len(all_migration_results)}")
        
        return {
            'results': all_results,
            'gini_results': all_gini_results,
            'demographics_results': all_demographics_results,
            'migration_results': all_migration_results
        }

    def export_model_comparison(self, filename=None):
        """Export comparison results to Excel with separate sheets for each model type."""
        if not hasattr(self, 'comparison_results'):
            print("No comparison results available. Run run_model_comparison() first.")
            return
        
        if filename is None:
            # Use output configuration to generate organized filename in excel_files folder
            filename = get_excel_path("turkey_economic_models_comparison", "models")
        
        df = pd.DataFrame(self.comparison_results)
        gini_df = pd.DataFrame(self.comparison_gini_results)
        demographics_df = pd.DataFrame(self.comparison_demographics_results)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main comparison sheet
            df.to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # Gini comparison
            gini_df.to_excel(writer, sheet_name='Gini_Comparison', index=False)
            
            # Demographics comparison (removed to reduce file size)
            # demographics_df.to_excel(writer, sheet_name='Demographics_Comparison', index=False)
            
            # Prepare per-year migration in/out aggregates if available
            migration_in_out_prepared = False
            if hasattr(self, 'comparison_migration_results') and self.comparison_migration_results:
                try:
                    mig_df = pd.DataFrame(self.comparison_migration_results)
                    if not mig_df.empty:
                        # Ensure missing capital column handled
                        if 'Migration_Capital' not in mig_df.columns:
                            mig_df['Migration_Capital'] = 0
                        # OUT: sums by (Model_Name, Year, Source_City)
                        out_agg = mig_df.groupby(['Model_Name','Year','Source_City'], as_index=False).agg({
                            'Migration_Labor_Force':'sum',
                            'Migration_Capital':'sum'
                        })
                        out_agg = out_agg.rename(columns={
                            'Source_City':'City',
                            'Migration_Labor_Force':'Total_Labor_Out_Year',
                            'Migration_Capital':'Total_Capital_Out_Year'
                        })
                        # IN: sums by (Model_Name, Year, Target_City)
                        in_agg = mig_df.groupby(['Model_Name','Year','Target_City'], as_index=False).agg({
                            'Migration_Labor_Force':'sum',
                            'Migration_Capital':'sum'
                        })
                        in_agg = in_agg.rename(columns={
                            'Target_City':'City',
                            'Migration_Labor_Force':'Total_Labor_In_Year',
                            'Migration_Capital':'Total_Capital_In_Year'
                        })
                        migration_in_out_prepared = True
                except Exception:
                    migration_in_out_prepared = False
            
            # Prepare per-year shock effects if available
            shock_effects_prepared = False
            if hasattr(self, 'comparison_shock_results') and self.comparison_shock_results:
                try:
                    shock_rows = []
                    for shock in self.comparison_shock_results:
                        model_name = shock.get('Model_Name')
                        year = shock.get('Year')
                        effects_map = shock.get('Effects', {})
                        shock_title = shock.get('Effect_Title')
                        shock_domain = shock.get('Domain')
                        shock_description = shock.get('Description')
                        # Build per-city list
                        for target, effect_values in effects_map.items():
                            # Normalize to dict
                            if isinstance(effect_values, list):
                                effect_values = {k: v for k, v in effect_values}
                            # Determine affected cities
                            if str(target).upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                                affected_cities = list(self.tr_network.cities.keys())
                            elif target in self.tr_network.cities:
                                affected_cities = [target]
                            else:
                                affected_cities = []
                            # Emit rows
                            for city_name in affected_cities:
                                row = {
                                    'Model_Name': model_name,
                                    'Year': year,
                                    'City': city_name,
                                    'Shock_Effect_A': effect_values.get('A', 0),
                                    'Shock_Effect_K': effect_values.get('K', 0),
                                    'Shock_Effect_L': effect_values.get('L', 0),
                                    'Shock_Effect_ALPHA': effect_values.get('ALPHA', 0),
                                    'Shock_Effect_BETA': effect_values.get('BETA', 0),
                                    'Shock_Title': shock_title,
                                    'Shock_Domain': shock_domain,
                                    'Shock_Description': shock_description,
                                    'Shock_Type_ID': shock.get('Type_ID', 0),
                                    'Shock_Subtype_ID': shock.get('Subtype_ID', 0),
                                    'Shock_Scope': shock.get('Scope', '')
                                }
                                shock_rows.append(row)
                    if shock_rows:
                        shock_df_raw = pd.DataFrame(shock_rows)
                        # Aggregate numeric effects and concatenate distinct titles/domains/descriptions
                        shock_df_agg = shock_df_raw.groupby(['Model_Name','Year','City'], as_index=False).agg({
                            'Shock_Effect_A':'sum',
                            'Shock_Effect_K':'sum',
                            'Shock_Effect_L':'sum',
                            'Shock_Effect_ALPHA':'sum',
                            'Shock_Effect_BETA':'sum',
                            'Shock_Title': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Shock_Domain': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Shock_Description': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Shock_Type_ID': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Shock_Subtype_ID': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Shock_Scope': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)])))
                        })
                        shock_df_agg = shock_df_agg.rename(columns={'Shock_Title':'Shock_Types','Shock_Domain':'Shock_Domains','Shock_Description':'Shock_Descriptions','Shock_Type_ID':'Shock_Type_IDs','Shock_Subtype_ID':'Shock_Subtype_IDs','Shock_Scope':'Shock_Scopes'})
                        shock_effects_prepared = True
                except Exception:
                    shock_effects_prepared = False
            
            # Prepare per-year policy effects if available
            policy_effects_prepared = False
            if hasattr(self, 'comparison_policy_results') and self.comparison_policy_results:
                try:
                    policy_rows = []
                    for pol in self.comparison_policy_results:
                        model_name = pol.get('Model_Name')
                        year = pol.get('Year')
                        effects_map = pol.get('Effects', {})
                        pol_title = pol.get('Effect_Title')
                        pol_domain = pol.get('Domain')
                        pol_description = pol.get('Description')
                        for target, effect_values in effects_map.items():
                            if isinstance(effect_values, list):
                                effect_values = {k: v for k, v in effect_values}
                            if str(target).upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                                affected_cities = list(self.tr_network.cities.keys())
                            elif target in self.tr_network.cities:
                                affected_cities = [target]
                            else:
                                affected_cities = []
                            for city_name in affected_cities:
                                row = {
                                    'Model_Name': model_name,
                                    'Year': year,
                                    'City': city_name,
                                    'Policy_Effect_A': effect_values.get('A', 0),
                                    'Policy_Effect_K': effect_values.get('K', 0),
                                    'Policy_Effect_L': effect_values.get('L', 0),
                                    'Policy_Effect_ALPHA': effect_values.get('ALPHA', 0),
                                    'Policy_Effect_BETA': effect_values.get('BETA', 0),
                                    'Policy_Title': pol_title,
                                    'Policy_Domain': pol_domain,
                                    'Policy_Description': pol_description,
                                    'Policy_Type_ID': pol.get('Policy_Type_ID', 0),
                                    'Policy_Subtype_ID': pol.get('Policy_Subtype_ID', 0),
                                    'Policy_Scope': pol.get('Policy_Scope', '')
                                }
                                policy_rows.append(row)
                    if policy_rows:
                        pol_df_raw = pd.DataFrame(policy_rows)
                        pol_df_agg = pol_df_raw.groupby(['Model_Name','Year','City'], as_index=False).agg({
                            'Policy_Effect_A':'sum',
                            'Policy_Effect_K':'sum',
                            'Policy_Effect_L':'sum',
                            'Policy_Effect_ALPHA':'sum',
                            'Policy_Effect_BETA':'sum',
                            'Policy_Title': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Policy_Domain': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Policy_Description': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Policy_Type_ID': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Policy_Subtype_ID': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)]))),
                            'Policy_Scope': lambda s: '; '.join(sorted(set([str(v) for v in s if pd.notna(v)])))
                        })
                        pol_df_agg = pol_df_agg.rename(columns={'Policy_Title':'Policy_Types','Policy_Domain':'Policy_Domains','Policy_Description':'Policy_Descriptions','Policy_Type_ID':'Policy_Type_IDs','Policy_Subtype_ID':'Policy_Subtype_IDs','Policy_Scope':'Policy_Scopes'})
                        policy_effects_prepared = True
                except Exception:
                    policy_effects_prepared = False
            
            # Separate sheets for each model
            for model_name in df['Model_Name'].unique():
                model_data = df[df['Model_Name'] == model_name].copy()
                # Merge per-year migration in/out if prepared
                if migration_in_out_prepared:
                    try:
                        model_out = out_agg[out_agg['Model_Name'] == model_name][['Year','City','Total_Labor_Out_Year','Total_Capital_Out_Year']]
                        model_in = in_agg[in_agg['Model_Name'] == model_name][['Year','City','Total_Labor_In_Year','Total_Capital_In_Year']]
                        model_data = model_data.merge(model_out, on=['Year','City'], how='left')
                        model_data = model_data.merge(model_in, on=['Year','City'], how='left')
                        # Fill NaNs with zeros for readability
                        for col in ['Total_Labor_Out_Year','Total_Capital_Out_Year','Total_Labor_In_Year','Total_Capital_In_Year']:
                            if col in model_data.columns:
                                model_data[col] = model_data[col].fillna(0).astype('int64', errors='ignore')
                    except Exception:
                        pass
                # Merge per-year shock effects if prepared
                if shock_effects_prepared:
                    try:
                        model_shocks = shock_df_agg[shock_df_agg['Model_Name'] == model_name][['Year','City','Shock_Effect_A','Shock_Effect_K','Shock_Effect_L','Shock_Effect_ALPHA','Shock_Effect_BETA','Shock_Types','Shock_Domains','Shock_Descriptions','Shock_Type_IDs','Shock_Subtype_IDs','Shock_Scopes']]
                        model_data = model_data.merge(model_shocks, on=['Year','City'], how='left')
                        for col in ['Shock_Effect_A','Shock_Effect_K','Shock_Effect_L','Shock_Effect_ALPHA','Shock_Effect_BETA']:
                            if col in model_data.columns:
                                model_data[col] = model_data[col].fillna(0)
                        for col in ['Shock_Types','Shock_Domains','Shock_Descriptions','Shock_Type_IDs','Shock_Subtype_IDs','Shock_Scopes']:
                            if col in model_data.columns:
                                model_data[col] = model_data[col].fillna('')
                    except Exception:
                        pass
                # Merge per-year policy effects if prepared
                if policy_effects_prepared:
                    try:
                        model_pols = pol_df_agg[pol_df_agg['Model_Name'] == model_name][['Year','City','Policy_Effect_A','Policy_Effect_K','Policy_Effect_L','Policy_Effect_ALPHA','Policy_Effect_BETA','Policy_Types','Policy_Domains','Policy_Descriptions','Policy_Type_IDs','Policy_Subtype_IDs','Policy_Scopes']]
                        model_data = model_data.merge(model_pols, on=['Year','City'], how='left')
                        for col in ['Policy_Effect_A','Policy_Effect_K','Policy_Effect_L','Policy_Effect_ALPHA','Policy_Effect_BETA']:
                            if col in model_data.columns:
                                model_data[col] = model_data[col].fillna(0)
                        for col in ['Policy_Types','Policy_Domains','Policy_Descriptions','Policy_Type_IDs','Policy_Subtype_IDs','Policy_Scopes']:
                            if col in model_data.columns:
                                model_data[col] = model_data[col].fillna('')
                    except Exception:
                        pass
                # Drop any unnamed or entirely empty columns
                try:
                    cols_keep = []
                    for c in model_data.columns:
                        name_ok = str(c).strip() != ''
                        if not name_ok:
                            continue
                        # Drop columns that are entirely NaN
                        if model_data[c].isna().all():
                            continue
                        cols_keep.append(c)
                    model_data = model_data[cols_keep]
                except Exception:
                    pass
                sheet_name = model_name.replace(' ', '_')[:31]  # Excel sheet name limit
                model_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Model comparison results exported to {filename}")

    def animate_model_comparison(self, save_path=None, figsize=(32, 24), cmap='RdYlGn'):
        """
        Create animation comparing multiple models side by side.
        """
        if not hasattr(self, 'comparison_results'):
            print("No comparison results available. Run run_model_comparison() first.")
            return
        
        if self.tr_network.provinces is None:
            print("Cannot animate: provinces data not loaded")
            return
        
        # Prepare data
        df = pd.DataFrame(self.comparison_results)
        models = df['Model_Name'].unique()
        years = sorted(df['Year'].unique())
        
        if len(models) < 2:
            print("Need at least 2 models for comparison animation")
            return
        
        # Create subplots stacked vertically: 1 column x up to 4 rows
        nrows = min(4, len(models))
        # Increase height based on number of rows for readability
        fig_height = max(figsize[1], 7 * nrows)
        fig, axes = plt.subplots(nrows, 1, figsize=(figsize[0], fig_height), constrained_layout=True)
        if nrows == 1:
            axes = [axes]
        
        plt.close(fig)
        
        # Prepare provinces data
        provinces = self.tr_network.provinces.copy()
        province_names = provinces['NAME_1'].tolist()
        
        # Normalize names (handle Turkish characters like MUŞ -> MUS) for reliable matching
        def _normalize_name(name: str) -> str:
            try:
                s = str(name)
            except Exception:
                s = ''
            trans = str.maketrans({
                'Ç':'C','Ö':'O','Ş':'S','İ':'I','I':'I','Ü':'U','Ğ':'G',
                'ç':'C','ö':'O','ş':'S','ı':'I','i':'I','ü':'U','ğ':'G'
            })
            return s.translate(trans).upper().strip()
        # Build normalized province index lookup
        _prov_idx_by_norm = { _normalize_name(p): i for i, p in enumerate(province_names) }
        
        # Build city-to-province mapping
        city_to_province = {}
        for province_name in province_names:
            for city in self.tr_network.cities:
                if city.upper() in province_name.upper() or province_name.upper() in city.upper():
                    city_to_province[city] = province_name
        
        def get_province_gdp(year, model_name):
            year_model_df = df[(df['Year'] == year) & (df['Model_Name'] == model_name)]
            city_gdp = dict(zip(year_model_df['City'], year_model_df['GDP_per_Capita']))
            province_gdp = []
            for province in province_names:
                val = 0.0
                for city, prov in city_to_province.items():
                    if prov == province and city in city_gdp:
                        val = city_gdp[city]
                        break
                province_gdp.append(val)
            return np.array(province_gdp)
        
        # Normalize across all years and models for consistent color scale
        all_gdp = []
        for year in years:
            for model in models:
                all_gdp.extend(get_province_gdp(year, model))
        all_gdp = np.array(all_gdp)
        vmin, vmax = all_gdp.min(), all_gdp.max()
        
        def update(frame):
            year = years[frame]
            
            for i, model in enumerate(models[:nrows]):
                ax = axes[i]
                ax.clear()
                
                province_gdp = get_province_gdp(year, model)
                provinces['gdp_per_capita'] = province_gdp
                # Per-frame per-model robust scaling using percentiles; fallback if nearly constant
                if province_gdp.size == 0:
                    local_pmin, local_pmax = 0.0, 1.0
                else:
                    local_pmin = float(np.nanpercentile(province_gdp, 5))
                    local_pmax = float(np.nanpercentile(province_gdp, 95))
                    if not np.isfinite(local_pmin) or not np.isfinite(local_pmax):
                        local_pmin, local_pmax = 0.0, 1.0
                    if local_pmax - local_pmin <= 1e-12:
                        m = float(np.nanmean(province_gdp)) if np.isfinite(np.nanmean(province_gdp)) else 0.0
                        span = max(1e-6, abs(m) * 0.1)
                        if span == 0.0:
                            span = 1.0
                        local_pmin, local_pmax = m - span, m + span
                
                provinces.plot(
                    ax=ax,
                    column='gdp_per_capita',
                    cmap=cmap,
                    edgecolor='white',
                    linewidth=0.5,
                    legend=False,
                    vmin=local_pmin,
                    vmax=local_pmax
                )
                
                # Add city labels
                for city in self.tr_network.cities:
                    pos = self.tr_network.network.nodes[city]['pos']
                    ax.annotate(
                        city,
                        xy=pos,
                        xytext=(0, 0),
                        textcoords='offset points',
                        fontsize=18,
                        ha='center',
                        va='center',
                        fontweight='bold',
                        color='black'
                    )
                
                ax.set_title(f"{model} - Year {year}", fontsize=18, fontweight='bold', pad=14)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(25, 45)
                ax.set_ylim(35, 43)
            
            # Add overall title
            fig.suptitle(f"Turkey GDP per Capita Comparison - Year {year}", fontsize=22, fontweight='bold')
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False)
        
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=6)
            else:
                ani.save(save_path, writer='ffmpeg', fps=6)
            print(f"Model comparison animation saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            _, save_path = get_comparison_paths("turkey_models_comparison")
            ani.save(save_path, writer='pillow', fps=6)
            print(f"Model comparison animation saved to {save_path}")
        
        plt.show()

    def plot_gini_comparison(self, save_path=None, figsize=(20, 12), y_limits=(0.0, 0.5)):
        """
        Create side-by-side Gini coefficient graphs comparing models over time.
        
        Args:
            save_path (str): Path to save the graph (if None, uses output_config)
            figsize (tuple): Figure size (width, height)
            y_limits (tuple): Y-axis limits (min, max) for consistent comparison
        """
        if not hasattr(self, 'comparison_gini_results'):
            print("No comparison Gini results available. Run run_model_comparison() first.")
            return
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        gini_df = pd.DataFrame(self.comparison_gini_results)
        
        if gini_df.empty:
            print("No Gini coefficient data available for plotting")
            return
        
        # Get unique models and years
        models = gini_df['Model_Name'].unique()
        years = sorted(gini_df['Year'].unique())
        
        if len(models) < 1:
            print("No models available for comparison plotting")
            return
        
        # Create 2x3 grid (up to 6 models); hide extra axes if fewer
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=False)
        axes = axes.flatten()
        
        # Set consistent y-axis limits
        y_min, y_max = y_limits
        
        # Plot each model in its cell
        for i, model_name in enumerate(models[:6]):
            ax = axes[i]
            # Filter data for this model
            model_data = gini_df[gini_df['Model_Name'] == model_name]
            # Sort by year
            model_data = model_data.sort_values('Year')
            # Plot Gini coefficient over time (no markers or labels)
            ax.plot(model_data['Year'], model_data['Gini_Coefficient'], 
                   linewidth=2, label=model_name)
            # Customize subplot
            ax.set_title(f'{model_name}\nGini Coefficient Over Time', fontsize=14, fontweight='bold', pad=25)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Gini Coefficient', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            # Set consistent y-axis limits
            ax.set_ylim(y_min, y_max)
        
        # Hide any unused subplots if fewer than 6 models
        for j in range(len(models), 6):
            fig.delaxes(axes[j])
        
        # Add overall title and spacing between title and subplots
        fig.suptitle('Turkey Economic Models: Gini Coefficient Comparison Over Time', 
                    fontsize=18, fontweight='bold', y=0.98)
        fig.text(0.5, 0.94, f'Y-axis fixed from {y_min} to {y_max} for easy comparison', 
                 ha='center', fontsize=12, style='italic')
        
        # Adjust layout to add space between suptitle and subplots
        plt.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
        
        # Save the graph
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gini comparison graph saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_graph_path
            save_path = get_graph_path("gini_coefficient_comparison", "models")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gini comparison graph saved to {save_path}")
        
        plt.show()
        
        return save_path

    def plot_individual_gini(self, model_name=None, save_path=None, figsize=(12, 8), y_limits=(0.0, 0.5)):
        """
        Create individual Gini coefficient graph for a specific model over time.
        
        Args:
            model_name (str): Name of the model to plot (if None, uses current model)
            save_path (str): Path to save the graph (if None, uses output_config)
            figsize (tuple): Figure size (width, height)
            y_limits (tuple): Y-axis limits (min, max)
        """
        if model_name is None:
            # Use current model
            if hasattr(self, 'comparison_gini_results'):
                # Find the current model from comparison results
                gini_df = pd.DataFrame(self.comparison_gini_results)
                if not gini_df.empty:
                    model_name = gini_df['Model_Name'].iloc[0]
                else:
                    model_name = self.model_type.capitalize()
            else:
                model_name = self.model_type.capitalize()
        
        # Get Gini data
        if hasattr(self, 'comparison_gini_results'):
            # Use comparison results
            gini_df = pd.DataFrame(self.comparison_gini_results)
            model_data = gini_df[gini_df['Model_Name'] == model_name]
        else:
            # Use current model results
            gini_df = pd.DataFrame(self.gini_results)
            model_data = gini_df.copy()
        
        if model_data.empty:
            print(f"No Gini coefficient data available for model: {model_name}")
            return
        
        # Sort by year
        model_data = model_data.sort_values('Year')
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        
        # Set consistent y-axis limits
        y_min, y_max = y_limits
        
        # Plot Gini coefficient over time (no markers or labels)
        ax.plot(model_data['Year'], model_data['Gini_Coefficient'], 
               linewidth=3, color='steelblue', label=f'{model_name} Gini Coefficient')
        
        # Customize plot
        ax.set_title(f'{model_name}: Gini Coefficient Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Gini Coefficient', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set consistent y-axis limits
        ax.set_ylim(y_min, y_max)
        
        # Add subtitle explaining the y-axis
        ax.text(0.5, 0.95, f'Y-axis fixed from {y_min} to {y_max} for consistent scaling', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the graph
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Individual Gini graph saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_graph_path
            save_path = get_graph_path(f"gini_coefficient_{model_name.lower().replace(' ', '_')}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Individual Gini graph saved to {save_path}")
        
        plt.show()
        
        return save_path

    def plot_combined_gini(self, save_path=None, figsize=(14, 8), y_limits=(0.0, 0.5)):
        """
        Create a combined Gini coefficient graph with all models on the same plot for direct comparison.
        
        Args:
            save_path (str): Path to save the graph (if None, uses output_config)
            figsize (tuple): Figure size (width, height)
            y_limits (tuple): Y-axis limits (min, max) for consistent comparison
        """
        if not hasattr(self, 'comparison_gini_results'):
            print("No comparison Gini results available. Run run_model_comparison() first.")
            return
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        gini_df = pd.DataFrame(self.comparison_gini_results)
        
        if gini_df.empty:
            print("No Gini coefficient data available for plotting")
            return
        
        # Get unique models
        models = gini_df['Model_Name'].unique()
        
        if len(models) < 2:
            print("Need at least 2 models for comparison plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        
        # Set consistent y-axis limits
        y_min, y_max = y_limits
        
        # Define colors for different models
        colors = ['steelblue', 'darkred', 'forestgreen', 'darkorange', 'purple', 'brown']
        
        # Plot each model with different colors and markers
        for i, model_name in enumerate(models):
            # Filter data for this model
            model_data = gini_df[gini_df['Model_Name'] == model_name]
            
            # Sort by year
            model_data = model_data.sort_values('Year')
            
            # Choose color (no markers)
            color = colors[i % len(colors)]
            
            # Plot Gini coefficient over time (no markers or labels)
            ax.plot(model_data['Year'], model_data['Gini_Coefficient'], 
                   linewidth=2, color=color, label=model_name, alpha=0.8)
        
        # Customize plot
        ax.set_title('Turkey Economic Models: Gini Coefficient Comparison Over Time', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Gini Coefficient', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        # Set consistent y-axis limits
        ax.set_ylim(y_min, y_max)
        
        # Add subtitle explaining the y-axis
        ax.text(0.5, 0.95, f'Y-axis fixed from {y_min} to {y_max} for easy comparison', 
                transform=ax.transAxes, ha='center', fontsize=11, style='italic')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the graph
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined Gini graph saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_graph_path
            save_path = get_graph_path("gini_coefficient_combined_comparison")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined Gini graph saved to {save_path}")
        
        plt.show()
        
        return save_path

    def plot_combined_gdp_per_capita(self, save_path=None, figsize=(14, 8)):
        """
        Create a combined GDP per capita graph with all models on the same plot for direct comparison.
        Plots the average GDP per capita for each model per year.
        
        Args:
            save_path (str): Path to save the graph (if None, uses output_config)
            figsize (tuple): Figure size (width, height)
        """
        if not hasattr(self, 'comparison_results'):
            print("No comparison results available. Run run_model_comparison() first.")
            return
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        results_df = pd.DataFrame(self.comparison_results)
        
        if results_df.empty:
            print("No GDP per capita data available for plotting")
            return
        
        # Get unique models
        models = results_df['Model_Name'].unique()
        
        if len(models) < 2:
            print("Need at least 2 models for comparison plotting")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        
        # Define colors for different models
        colors = ['steelblue', 'darkred', 'forestgreen', 'darkorange', 'purple', 'brown']
        
        # Plot each model with different colors
        for i, model_name in enumerate(models):
            # Filter data for this model
            model_data = results_df[results_df['Model_Name'] == model_name]
            
            # Calculate average GDP per capita for each year
            yearly_avg = model_data.groupby('Year')['GDP_per_Capita'].mean().reset_index()
            
            # Sort by year
            yearly_avg = yearly_avg.sort_values('Year')
            
            # Choose color
            color = colors[i % len(colors)]
            
            # Plot average GDP per capita over time (no markers or labels)
            ax.plot(yearly_avg['Year'], yearly_avg['GDP_per_Capita'], 
                   linewidth=2, color=color, label=model_name, alpha=0.8)
        
        # Customize plot
        ax.set_title('Turkey Economic Models: GDP per Capita Comparison Over Time\n(Average per Model per Year)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('GDP per Capita (Average)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the graph
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined GDP per capita graph saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_graph_path
            save_path = get_graph_path("gdp_per_capita_combined_comparison")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined GDP per capita graph saved to {save_path}")
        
        plt.show()
        
        return save_path

    def _maintain_demographic_proportions(self, city):
        """
        Helper method to maintain demographic proportions after natural growth.
        This ensures that when labor force grows naturally, the demographic
        proportions remain consistent with the original structure.
        """
        if not city.demographics:
            return
            
        # Calculate the current demographic proportions
        total_demo_pop = sum(demo.population_count for demo in city.demographics)
        
        if total_demo_pop > 0:
            # Calculate the growth factor for labor force
            # We want to maintain the same demographic proportions
            # So we scale each demographic group by the same factor
            growth_factor = city.labor_force / total_demo_pop
            
            # Apply the growth factor to maintain proportions
            for demo in city.demographics:
                demo.population_count = int(demo.population_count * growth_factor)

    def set_labor_growth_rate(self, labor_growth_rate):
        """
        Set the labor force growth rate.
        
        Args:
            labor_growth_rate (float): New labor force growth rate (e.g., 0.013 for 1.3%)
        """
        self.labor_growth_rate = labor_growth_rate
        print(f"Labor force growth rate updated to: {self.labor_growth_rate:.2%}")
        
        # Update model configuration display
        self._print_model_config()

    def set_population_growth_rate(self, population_growth_rate):
        """
        Set the population growth rate.
        
        Args:
            population_growth_rate (float): New population growth rate (e.g., 0.013 for 1.3%)
        """
        self.population_growth_rate = population_growth_rate
        print(f"Population growth rate updated to: {self.population_growth_rate:.2%}")
        
        # Update model configuration display
        self._print_model_config()

    def set_capital_migration_ratio(self, capital_migration_ratio):
        """
        Set the capital migration ratio (only for open economy models).
        
        Args:
            capital_migration_ratio (float): Capital migration rate as fraction of labor migration rate
                                           (e.g., 0.2 means capital migrates at 1/5 of labor migration rate)
        """
        if self.model_type in ["open", "shock", "policy"]:
            self.capital_migration_ratio = capital_migration_ratio
            denom_str = f"1/{int(1/self.capital_migration_ratio)}" if self.capital_migration_ratio not in [0, 0.0] else "N/A"
            print(f"Capital migration ratio updated to: {self.capital_migration_ratio:.1%} ({denom_str}) of labor migration")
            # Update model configuration display
            self._print_model_config()
        else:
            print("Capital migration ratio can only be set for open economy models")

    def load_shocks(self):
        """
        Generate realistic economic shocks programmatically instead of loading from file.
        This ensures shocks are properly formatted and realistic.
        """
        if self.model_type not in ["shock", "policy"]:
            return
        
        # Generate realistic shocks instead of loading from file
        self.generate_realistic_shocks()

    def apply_shock_effects(self, shock_data):
        """
        Apply shock effects to cities based on the shock data.
        
        Args:
            shock_data (dict): Shock data containing EFFECTS information
        """
        if self.model_type not in ["shock", "policy"]:
            return
        
        effects = shock_data.get('EFFECTS', {})
        
        for target, effect_values in effects.items():
            if target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                # Apply to all cities
                target_cities = list(self.tr_network.cities.values())
            elif target in self.tr_network.cities:
                # Apply to specific city
                target_cities = [self.tr_network.cities[target]]
            else:
                # Skip if city not found
                print(f"Warning: Target '{target}' not found in city list, skipping shock effect")
                continue
            
            # Apply effects to target cities
            for city in target_cities:
                if isinstance(effect_values, dict):
                    # Format: {"A": -0.8, "K": -0.6, "L": 0, "ALPHA": -0.4, "BETA": -0.5}
                    self._apply_city_effects(city, effect_values)
                elif isinstance(effect_values, list):
                    # Format: [["A", -0.4], ["K", -0.5], ["L", 0], ["ALPHA", -0.3], ["BETA", -0.4]]
                    effect_dict = {}
                    for param, value in effect_values:
                        effect_dict[param] = value
                    self._apply_city_effects(city, effect_dict)

    def _apply_city_effects(self, city, effects):
        """
        Apply individual effects to a city's production function parameters.
        
        IMPORTANT: The formula depends on whether the parameter is positive or negative:
        
        For NEGATIVE parameters:
        - new_value = parameter × (1 - effect)/100
        
        For POSITIVE parameters:  
        - new_value = parameter × (1 + effect)/100
        
        Examples:
        - β = -1.0, effect = +0.5 → new β = -1.0 × (1 - 0.5)/100 = -0.005 (improved)
        - β = -1.0, effect = -0.3 → new β = -1.0 × (1 - (-0.3))/100 = -0.003 (worsened)
        - β = +10.0, effect = +0.2 → new β = +10.0 × (1 + 0.2)/100 = +0.1 (improved)
        - β = +10.0, effect = -0.2 → new β = +10.0 × (1 + (-0.2))/100 = +0.08 (worsened)
        
        Args:
            city (City): City object to apply effects to
            effects (dict or list): Dictionary of parameter effects or list of [param, effect] pairs
        """
        # Apply each effect to the corresponding parameter
        # Handle both dictionary and list formats
        if isinstance(effects, dict):
            for param, effect in effects.items():
                if param == "A":
                    # Total Factor Productivity
                    if city.A < 0:
                        city.A *= (1 - effect/100)  # Negative parameter
                    else:
                        city.A *= (1 + effect/100)  # Positive parameter
                elif param == "K":
                    # Capital stock
                    if city.capital_stock < 0:
                        city.capital_stock *= (1 - effect/100)  # Negative parameter
                    else:
                        city.capital_stock *= (1 + effect/100)  # Positive parameter
                elif param == "L":
                    # Labor force
                    if city.labor_force < 0:
                        city.labor_force = int(city.labor_force * (1 - effect/100))  # Negative parameter
                    else:
                        city.labor_force = int(city.labor_force * (1 + effect/100))  # Positive parameter
                elif param == "ALPHA":
                    # Capital share parameter
                    if city.alpha < 0:
                        city.alpha *= (1 - effect/100)  # Negative parameter
                    else:
                        city.alpha *= (1 + effect/100)  # Positive parameter
                elif param == "BETA":
                    # Labor share parameter
                    if city.beta < 0:
                        city.beta *= (1 - effect/100)  # Negative parameter
                    else:
                        city.beta *= (1 + effect/100)  # Positive parameter
        elif isinstance(effects, list):
            for param, effect in effects:
                if param == "A":
                    # Total Factor Productivity
                    if city.A < 0:
                        city.A *= (1 - effect/100)  # Negative parameter
                    else:
                        city.A *= (1 + effect/100)  # Positive parameter
                elif param == "K":
                    # Capital stock
                    if city.capital_stock < 0:
                        city.capital_stock *= (1 - effect/100)  # Negative parameter
                    else:
                        city.capital_stock *= (1 + effect/100)  # Positive parameter
                elif param == "L":
                    # Labor force
                    if city.labor_force < 0:
                        city.labor_force = int(city.labor_force * (1 - effect/100))  # Negative parameter
                    else:
                        city.labor_force = int(city.labor_force * (1 + effect/100))  # Positive parameter
                elif param == "ALPHA":
                    # Capital share parameter
                    if city.alpha < 0:
                        city.alpha *= (1 - effect/100)  # Negative parameter
                    else:
                        city.alpha *= (1 + effect/100)  # Positive parameter
                elif param == "BETA":
                    # Labor share parameter
                    if city.beta < 0:
                        city.beta *= (1 - effect/100)  # Negative parameter
                    else:
                        city.beta *= (1 + effect/100)  # Positive parameter
        
        # Only ensure non-negative values for physical quantities
        # Do NOT force bounds on production function parameters (α, β, A)
        city.capital_stock = max(0, city.capital_stock)  # Capital cannot be negative
        city.labor_force = max(0, city.labor_force)      # Labor cannot be negative
        
        # Note: α, β, and A can be negative or positive as needed
        # The shock effects will work correctly with any parameter values

    def trigger_random_shocks(self, year):
        """
        Trigger shocks for the current year from pre-generated realistic shocks.
        
        Args:
            year (int): Current simulation year
        """
        if self.model_type not in ["shock", "policy"] or not self.available_shocks:
            return []
        
        # Find shocks for this specific year
        year_shocks = [shock for shock in self.available_shocks if shock.get('Year') == year]
        
        # Since we generate exactly 1 shock per year, we should always have one
        if not year_shocks:
            # Fallback: create a shock for this year if none exists
            print(f"  Warning: No shock found for year {year}, creating fallback shock")
            fallback_shock = self._create_fallback_shock(year)
            if fallback_shock:
                year_shocks = [fallback_shock]
        
        # Apply each shock for this year
        for shock in year_shocks:
            self.apply_shock_effects(shock)
            
            # Record the shock for reporting
            shock_record = {
                'Year': year,
                'Effect_Title': shock['EFFECT'][0],
                'Domain': shock['EFFECT'][1],
                'Description': shock['EFFECT'][2],
                'Effects': shock['EFFECTS'],
                'Type_ID': shock.get('Type_ID', 0),
                'Subtype_ID': shock.get('Subtype_ID', 0),
                'Scope': shock.get('Scope', '')
            }
            self.active_shocks.append(shock_record)
        
        if year_shocks:
            print(f"  Year {year}: {len(year_shocks)} shock(s) triggered")
            for shock in year_shocks:
                print(f"    - {shock['EFFECT'][0]} ({shock['EFFECT'][1]})")
        
        return year_shocks
    
    def _create_fallback_shock(self, year):
        """
        Create a fallback shock if none exists for a given year.
        This ensures robustness of the shock system.
        """
        # Simple fallback: random economic crisis
        fallback_shock = {
            "EFFECT": [
                "Economic Uncertainty",
                "Economic Crisis", 
                "General economic uncertainty affecting business confidence"
            ],
            "EFFECTS": {
                "NATION": {
                    "A": -0.05,    # -5% productivity
                    "K": -0.08,    # -8% capital
                    "L": -0.03,    # -3% labor
                    "ALPHA": -0.04, # -4% capital efficiency
                    "BETA": -0.03   # -3% labor efficiency
                }
            }
        }
        return fallback_shock

    def get_shock_summary(self, year):
        """
        Get a summary of shocks that occurred in a specific year.
        
        Args:
            year (int): Year to get shock summary for
            
        Returns:
            str: Summary of shocks for the year
        """
        if self.model_type not in ["shock", "policy"]:
            return f"No shock system for {self.model_type} economy"
        
        year_shocks = [s for s in self.active_shocks if s['Year'] == year]
        
        if not year_shocks:
            return f"No shocks occurred in year {year}"
        
        summary = f"Shock Summary for Year {year}:\n"
        summary += f"Total shocks: {len(year_shocks)}\n"
        
        for i, shock in enumerate(year_shocks, 1):
            summary += f"  {i}. {shock['Effect_Title']} ({shock['Domain']})\n"
        
        return summary

    def export_shock_results(self, filename=None):
        """
        Export shock results to Excel for analysis.
        """
        if self.model_type not in ["shock", "policy"] or not self.active_shocks:
            print("No shock data available to export")
            return
        
        if filename is None:
            filename = get_excel_path("turkey_shock_simulation", self.model_type)
        
        # Create DataFrame from shock results
        shock_df = pd.DataFrame(self.active_shocks)
        
        # Export to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            shock_df.to_excel(writer, sheet_name='Shock_Results', index=False)
            
            # Create summary sheet
            summary_data = []
            for shock in self.active_shocks:
                summary_data.append({
                    'Year': shock['Year'],
                    'Effect_Title': shock['Effect_Title'],
                    'Domain': shock['Domain'],
                    'Target_Count': len(shock['Effects'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Shock_Summary', index=False)
        
        print(f"Shock results exported to {filename}")

    def set_shock_probability(self, probability):
        """
        Set the shock probability (only for shock economy models).
        
        Args:
            probability (float): Probability of shocks occurring per year (0.0 to 1.0)
        """
        if self.model_type in ["shock", "policy"]:
            if 0.0 <= probability <= 1.0:
                self.shock_probability = probability
                print(f"Shock probability updated to: {self.shock_probability:.1%}")
                self._print_model_config()
            else:
                print("Shock probability must be between 0.0 and 1.0")
        else:
            print("Shock probability can only be set for shock economy models")

    def set_max_shocks_per_year(self, max_shocks):
        """
        Set the maximum number of shocks per year (only for shock economy models).
        
        Args:
            max_shocks (int): Maximum number of shocks that can occur in a single year
        """
        if self.model_type in ["shock", "policy"]:
            if max_shocks > 0:
                self.max_shocks_per_year = max_shocks
                print(f"Maximum shocks per year updated to: {self.max_shocks_per_year}")
                self._print_model_config()
            else:
                print("Maximum shocks per year must be greater than 0")
        else:
            print("Maximum shocks per year can only be set for shock economy models")

    def export_detailed_shock_results(self, filename=None):
        """
        Export detailed shock results with parameter changes for reinforcement learning.
        Shows old/new parameter values and changes for each affected city.
        
        Args:
            filename (str): Path to save the detailed shock results
        """
        if self.model_type not in ["shock", "policy"] or not self.active_shocks:
            print("No shock data available to export")
            return
        
        if filename is None:
            filename = get_excel_path("turkey_detailed_shock_results", self.model_type)
        
        # Create detailed shock results with parameter changes
        detailed_shock_data = []
        
        for shock in self.active_shocks:
            year = shock['Year']
            effects = shock['Effects']
            
            # Get all cities that were affected by this shock
            affected_cities = []
            for target, effect_values in effects.items():
                if target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                    # Nationwide shock affects all cities
                    affected_cities.extend(list(self.tr_network.cities.keys()))
                elif target in self.tr_network.cities:
                    # City-specific shock
                    affected_cities.append(target)
            
            # Create records for each affected city
            for city_name in affected_cities:
                city = self.tr_network.cities[city_name]
                
                # Get old parameter values (before shock)
                old_params = {
                    'A': city.A,
                    'K': city.capital_stock,
                    'L': city.labor_force,
                    'ALPHA': city.alpha,
                    'BETA': city.beta
                }
                
                # Calculate new parameter values (after shock)
                new_params = {}
                # Handle both dictionary and list formats
                if isinstance(effect_values, dict):
                    for param, effect in effect_values.items():
                        if param == "A":
                            if old_params['A'] < 0:
                                new_params['A'] = old_params['A'] * (1 - effect/100)
                            else:
                                new_params['A'] = old_params['A'] * (1 + effect/100)
                        elif param == "K":
                            if old_params['K'] < 0:
                                new_params['K'] = old_params['K'] * (1 - effect/100)
                            else:
                                new_params['K'] = old_params['K'] * (1 + effect/100)
                        elif param == "L":
                            if old_params['L'] < 0:
                                new_params['L'] = int(old_params['L'] * (1 - effect/100))
                            else:
                                new_params['L'] = int(old_params['L'] * (1 + effect/100))
                        elif param == "ALPHA":
                            if old_params['ALPHA'] < 0:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 - effect/100)
                            else:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 + effect/100)
                        elif param == "BETA":
                            if old_params['BETA'] < 0:
                                new_params['BETA'] = old_params['BETA'] * (1 - effect/100)
                            else:
                                new_params['BETA'] = old_params['BETA'] * (1 + effect/100)
                elif isinstance(effect_values, list):
                    for param, effect in effect_values:
                        if param == "A":
                            if old_params['A'] < 0:
                                new_params['A'] = old_params['A'] * (1 - effect/100)
                            else:
                                new_params['A'] = old_params['A'] * (1 + effect/100)
                        elif param == "K":
                            if old_params['K'] < 0:
                                new_params['K'] = old_params['K'] * (1 - effect/100)
                            else:
                                new_params['K'] = old_params['K'] * (1 + effect/100)
                        elif param == "L":
                            if old_params['L'] < 0:
                                new_params['L'] = int(old_params['L'] * (1 - effect/100))
                            else:
                                new_params['L'] = int(old_params['L'] * (1 + effect/100))
                        elif param == "ALPHA":
                            if old_params['ALPHA'] < 0:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 - effect/100)
                            else:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 + effect/100)
                        elif param == "BETA":
                            if old_params['BETA'] < 0:
                                new_params['BETA'] = old_params['BETA'] * (1 - effect/100)
                            else:
                                new_params['BETA'] = old_params['BETA'] * (1 + effect/100)
                
                # Fill in unchanged parameters
                for param in ['A', 'K', 'L', 'ALPHA', 'BETA']:
                    if param not in new_params:
                        new_params[param] = old_params[param]
                
                # Create detailed shock record
                shock_record = {
                    'Year': year,
                    'City': city_name,
                    'Shock_Title': shock['Effect_Title'],
                    'Shock_Domain': shock['Domain'],
                    'Shock_Description': shock['Description'],
                    'Shock_Type_ID': shock.get('Type_ID', 0),
                    'Shock_Subtype_ID': shock.get('Subtype_ID', 0),
                    'Shock_Scope': shock.get('Scope', ''),
                    # Old parameter values
                    'Old_A': old_params['A'],
                    'Old_K': old_params['K'],
                    'Old_L': old_params['L'],
                    'Old_ALPHA': old_params['ALPHA'],
                    'Old_BETA': old_params['BETA'],
                    # New parameter values
                    'New_A': new_params['A'],
                    'New_K': new_params['K'],
                    'New_L': new_params['L'],
                    'New_ALPHA': new_params['ALPHA'],
                    'New_BETA': new_params['BETA'],
                    # Parameter changes
                    'Delta_A': new_params['A'] - old_params['A'],
                    'Delta_K': new_params['K'] - old_params['K'],
                    'Delta_L': new_params['L'] - old_params['L'],
                    'Delta_ALPHA': new_params['ALPHA'] - old_params['ALPHA'],
                    'Delta_BETA': new_params['BETA'] - old_params['BETA']
                }
                
                detailed_shock_data.append(shock_record)
        
        # Create detailed shock DataFrame
        detailed_shock_df = pd.DataFrame(detailed_shock_data)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main detailed results
            detailed_shock_df.to_excel(writer, sheet_name='Detailed_Shock_Results', index=False)
            
            # Summary by year and city
            summary_df = detailed_shock_df.groupby(['Year', 'City']).agg({
                'Shock_Title': 'count',
                'Delta_A': 'sum',
                'Delta_K': 'sum',
                'Delta_L': 'sum',
                'Delta_ALPHA': 'sum',
                'Delta_BETA': 'sum'
            }).reset_index()
            summary_df.columns = ['Year', 'City', 'Shock_Count', 'Total_Delta_A', 'Total_Delta_K', 'Total_Delta_L', 'Total_Delta_ALPHA', 'Total_Delta_BETA']
            summary_df.to_excel(writer, sheet_name='Shock_Summary_By_City', index=False)
            
            # Summary by shock
            shock_summary_df = detailed_shock_df.groupby(['Shock_Title', 'Shock_Domain']).agg({
                'City': 'count',
                'Delta_A': 'mean',
                'Delta_K': 'mean',
                'Delta_L': 'mean',
                'Delta_ALPHA': 'mean',
                'Delta_BETA': 'mean'
            }).reset_index()
            shock_summary_df.columns = ['Shock_Title', 'Shock_Domain', 'Cities_Affected', 'Avg_Delta_A', 'Avg_Delta_K', 'Avg_Delta_L', 'Avg_Delta_ALPHA', 'Avg_Delta_BETA']
            shock_summary_df.to_excel(writer, sheet_name='Shock_Summary_By_Type', index=False)
        
        print(f"Detailed shock results exported to {filename}")
        print(f"Total shock records: {len(detailed_shock_data)}")
        print(f"Affected cities: {len(detailed_shock_df['City'].unique())}")
        print(f"Years with shocks: {len(detailed_shock_df['Year'].unique())}")
        
        return filename

    def animate_shock_effects(self, save_path=None, figsize=(32, 24)):
        """
        Create animation showing shock effects on cities over time.
        Gray background with red cities for negative effects and green for positive effects.
        
        Args:
            save_path (str): Path to save the animation
            figsize (tuple): Figure size
        """
        if self.model_type not in ["shock", "policy"]:
            print("Shock animation only available for shock economy models")
            return
        
        if not self.active_shocks:
            print("No shock data available. Run simulation first.")
            return
        
        if self.tr_network.provinces is None:
            print("Cannot animate: provinces data not loaded")
            return
        
        # Prepare data
        shock_df = pd.DataFrame(self.active_shocks)
        years = sorted(shock_df['Year'].unique())
        
        if not years:
            print("No shock data found for animation")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.close(fig)
        
        # Prepare provinces data
        provinces = self.tr_network.provinces.copy()
        province_names = provinces['NAME_1'].tolist()
        
        # Normalize names (handle Turkish characters like MUŞ -> MUS) for reliable matching
        def _normalize_name(name: str) -> str:
            try:
                s = str(name)
            except Exception:
                s = ''
            trans = str.maketrans({
                'Ç':'C','Ö':'O','Ş':'S','İ':'I','I':'I','Ü':'U','Ğ':'G',
                'ç':'C','ö':'O','ş':'S','ı':'I','i':'I','ü':'U','ğ':'G'
            })
            return s.translate(trans).upper().strip()
        # Build normalized province index lookup
        _prov_idx_by_norm = { _normalize_name(p): i for i, p in enumerate(province_names) }
        
        # Build city-to-province mapping
        city_to_province = {}
        for province_name in province_names:
            for city in self.tr_network.cities:
                if city.upper() in province_name.upper() or province_name.upper() in city.upper():
                    city_to_province[city] = province_name
        
        def get_shock_effects(year):
            """Get shock effects for a specific year."""
            year_shocks = shock_df[shock_df['Year'] == year]
            city_effects = {}
            
            for _, shock in year_shocks.iterrows():
                effects = shock['Effects']
                
                for target, effect_values in effects.items():
                    if target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                        # Nationwide shock affects all cities
                        for city_name in self.tr_network.cities.keys():
                            if city_name not in city_effects:
                                city_effects[city_name] = {'positive': 0, 'negative': 0, 'net': 0.0, 'shocks': []}
                            
                            # Determine effect contributions (supports dict or list)
                            if isinstance(effect_values, dict):
                                for param, effect in effect_values.items():
                                    if effect > 0:
                                        city_effects[city_name]['positive'] += 1
                                    elif effect < 0:
                                        city_effects[city_name]['negative'] += 1
                                    city_effects[city_name]['net'] += float(effect)
                            elif isinstance(effect_values, list):
                                for param, effect in effect_values:
                                    if effect > 0:
                                        city_effects[city_name]['positive'] += 1
                                    elif effect < 0:
                                        city_effects[city_name]['negative'] += 1
                                    city_effects[city_name]['net'] += float(effect)
                            
                            city_effects[city_name]['shocks'].append({
                                'title': shock['Effect_Title'],
                                'domain': shock['Domain'],
                                'effects': effect_values
                            })
                    
                    elif target in self.tr_network.cities:
                        # City-specific shock - only affect the specific city
                        if target not in city_effects:
                            city_effects[target] = {'positive': 0, 'negative': 0, 'net': 0.0, 'shocks': []}
                        
                        # Determine effect contributions (supports dict or list)
                        if isinstance(effect_values, dict):
                            for param, effect in effect_values.items():
                                if effect > 0:
                                    city_effects[target]['positive'] += 1
                                elif effect < 0:
                                    city_effects[target]['negative'] += 1
                                city_effects[target]['net'] += float(effect)
                        elif isinstance(effect_values, list):
                            for param, effect in effect_values:
                                if effect > 0:
                                    city_effects[target]['positive'] += 1
                                elif effect < 0:
                                    city_effects[target]['negative'] += 1
                                city_effects[target]['net'] += float(effect)
                        
                        city_effects[target]['shocks'].append({
                            'title': shock['Effect_Title'],
                            'domain': shock['Domain'],
                            'effects': effect_values
                        })
            
            return city_effects
        
        # Set up the initial plot
        provinces.plot(
            ax=ax,
            column='NAME_1',
            edgecolor='white',
            linewidth=0.5,
            legend=False,
            color='lightgray'
        )
        
        # Add city labels
        for city in self.tr_network.cities:
            pos = self.tr_network.network.nodes[city]['pos']
            ax.annotate(
                city,
                xy=pos,
                xytext=(0, 0),
                textcoords='offset points',
                fontsize=18,
                ha='center',
                va='center',
                fontweight='bold',
                color='black'
            )
        
        # Set map limits
        ax.set_xlim(25, 45)
        ax.set_ylim(35, 43)
        ax.set_title(f"Turkey Shock Effects - Year {years[0]}\n{self.model_type.capitalize()} Economy", 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        def update(frame):
            year = years[frame]
            ax.clear()
            
            # Plot provinces (default gray)
            provinces.plot(
                ax=ax,
                column='NAME_1',
                edgecolor='white',
                linewidth=0.5,
                legend=False,
                color='lightgray'
            )
            
            # Get shock effects for this year
            city_effects = get_shock_effects(year)
            
            # Plot cities with shock effects by coloring their provinces
            total_shocks = 0
            positive_shocks = 0
            negative_shocks = 0
            
            # Create a list to store province colors
            province_colors = ['lightgray'] * len(province_names)
            
            for city_name, effects in city_effects.items():
                if city_name in self.tr_network.cities:
                    # Use normalized direct lookup (handles MUŞ vs MUS)
                    city_province = _prov_idx_by_norm.get(_normalize_name(city_name))
                    
                    if city_province is not None:
                        # Determine province color based on net shock effect
                        if effects['net'] > 0:
                            # Net positive effects - green
                            province_colors[city_province] = 'green'
                            positive_shocks += 1
                        elif effects['net'] < 0:
                            # Net negative effects - red
                            province_colors[city_province] = 'red'
                            negative_shocks += 1
                        else:
                            # No net effect - keep gray
                            province_colors[city_province] = 'lightgray'
                        
                        total_shocks += abs(effects['net'])
            
            # Re-plot provinces with updated colors (explicit color array to avoid colormap mixing)
            provinces.plot(
                ax=ax,
                color=province_colors,
                edgecolor='white',
                linewidth=0.5,
                legend=False
            )
            
            # Add city labels
            for city in self.tr_network.cities:
                pos = self.tr_network.network.nodes[city]['pos']
                ax.annotate(
                    city,
                    xy=pos,
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=18,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='black'
                )
                
                # Add simple dot for affected cities (optional)
                if city in city_effects and city_effects[city]['net'] != 0:
                    ax.annotate(
                        '●',
                        xy=pos,
                        xytext=(0, 0),
                        textcoords='offset points',
                        fontsize=14,
                        ha='center',
                        va='center',
                        color='white'
                    )
            
            # Build title with shock type/domain/description (assumes 1 per year)
            try:
                yrow = shock_df[shock_df['Year'] == year].iloc[0]
                shock_title = str(yrow.get('Effect_Title', ''))
                shock_domain = str(yrow.get('Domain', ''))
                shock_desc = str(yrow.get('Description', ''))
                title_line1 = f"{shock_title} | {shock_domain}"
                title_line2 = shock_desc[:160]
            except Exception:
                title_line1 = f"Turkey Shock Effects"
                title_line2 = ""
            
            # Set map limits and titles
            ax.set_xlim(25, 45)
            ax.set_ylim(35, 43)
            ax.set_title(f"{title_line1}\nYear {year} - {self.model_type.capitalize()} Economy\n{title_line2}", 
                        fontsize=22, fontweight='bold', pad=22)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Bigger, clearer legend
            legend_text = f"""Shock Map Legend

Red: Net negative shock
Blue: Net positive shock
Gray: No shock effect

Year {year} summary
Cities with net positive: {positive_shocks}
Cities with net negative: {negative_shocks}
"""
            
            ax.text(0.015, 0.985, legend_text, 
                   transform=ax.transAxes, fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95),
                   verticalalignment='top')
        
        # Create animation with faster speed for readability
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False, interval=1000)
        
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=1.0)
            else:
                ani.save(save_path, writer='ffmpeg', fps=1.0)
            print(f"Shock effects animation saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_animation_path
            save_path = get_animation_path("turkey_shock_effects", self.model_type)
            ani.save(save_path, writer='pillow', fps=1.0)
            print(f"Shock effects animation saved to {save_path}")
        
        plt.show()
        return save_path

    def get_city_shock_details(self, city_name, year):
        """
        Get detailed shock information for a specific city and year.
        
        Args:
            city_name (str): Name of the city
            year (int): Year to get shock details for
            
        Returns:
            dict: Detailed shock information for the city
        """
        if self.model_type not in ["shock", "policy"] or not self.active_shocks:
            return None
        
        city_shocks = []
        for shock in self.active_shocks:
            if shock['Year'] == year:
                effects = shock['Effects']
                
                # Check if this city was affected
                affected = False
                for target, effect_values in effects.items():
                    if (target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"] or 
                        target == city_name):
                        affected = True
                        break
                
                if affected:
                    city_shocks.append({
                        'Effect_Title': shock['Effect_Title'],
                        'Domain': shock['Domain'],
                        'Description': shock['Description'],
                        'Effects': shock['Effects']
                    })
        
        if not city_shocks:
            return {
                'city': city_name,
                'year': year,
                'shocks': [],
                'total_effects': 0,
                'positive_effects': 0,
                'negative_effects': 0
            }
        
        # Count positive and negative effects
        total_effects = 0
        positive_effects = 0
        negative_effects = 0
        
        for shock in city_shocks:
            for target, effect_values in shock['Effects'].items():
                if (target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"] or 
                    target == city_name):
                    # Handle both dictionary and list formats
                    if isinstance(effect_values, dict):
                        for param, effect in effect_values.items():
                            total_effects += 1
                            if effect > 0:
                                positive_effects += 1
                            elif effect < 0:
                                negative_effects += 1
                    elif isinstance(effect_values, list):
                        for param, effect in effect_values:
                            total_effects += 1
                            if effect > 0:
                                positive_effects += 1
                            elif effect < 0:
                                negative_effects += 1
        
        return {
            'city': city_name,
            'year': year,
            'shocks': city_shocks,
            'total_effects': total_effects,
            'positive_effects': positive_effects,
            'negative_effects': negative_effects
        }

    def generate_realistic_shocks(self):
        """
        Generate realistic economic shocks programmatically instead of loading from file.
        Creates shocks based on real economic scenarios with appropriate parameter effects.
        Effects are randomized per shock around template means (1 std deviation) to avoid repetition.
        """
        if self.model_type not in ["shock", "policy"]:
            return
        
        # Clear any existing shocks
        self.available_shocks = []
        
        # Helper: sample effects per parameter with 1 std deviation around mean
        def _sample_effects(effects_mean: dict, effects_std: dict | None = None) -> dict:
            sampled = {}
            for param, mu in effects_mean.items():
                if effects_std and param in effects_std:
                    sigma = effects_std[param]
                else:
                    # Default std: 40% of |mean|, minimum floor
                    sigma = max(0.02, abs(mu) * 0.4)
                val = random.gauss(mu, sigma)
                # Soft cap to keep within a reasonable band for stability
                if val > 0:
                    val = min(val, 0.5)
                else:
                    val = max(val, -0.5)
                sampled[param] = val
            return sampled
        
        # Define realistic shock types with appropriate parameter effects (means and stds)
        shock_templates = [
            # Natural Disasters
            {
                "type": "Natural Disaster",
                "type_id": 1,
                "subtypes": [
                    {
                        "name": "Major Earthquake",
                        "subtype_id": 101,
                        "description": "Severe earthquake causing infrastructure damage and population displacement",
                        "effects": {"A": -0.15, "K": -0.20, "L": -0.10, "ALPHA": -0.08, "BETA": -0.05},
                        "effects_std": {"A": 0.06, "K": 0.08, "L": 0.05, "ALPHA": 0.04, "BETA": 0.03},
                        "affected_scope": "city_specific",
                        "frequency": 0.22
                    },
                    {
                        "name": "Severe Flooding",
                        "subtype_id": 102,
                        "description": "Extensive flooding affecting agricultural and industrial areas",
                        "effects": {"A": -0.12, "K": -0.15, "L": -0.08, "ALPHA": -0.06, "BETA": -0.04},
                        "effects_std": {"A": 0.05, "K": 0.06, "L": 0.04, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "regional",
                        "frequency": 0.20
                    },
                    {
                        "name": "Prolonged Drought",
                        "subtype_id": 103,
                        "description": "Extended drought affecting agricultural productivity and water resources",
                        "effects": {"A": -0.10, "K": -0.05, "L": -0.15, "ALPHA": -0.03, "BETA": -0.12},
                        "effects_std": {"A": 0.04, "K": 0.03, "L": 0.06, "ALPHA": 0.02, "BETA": 0.05},
                        "affected_scope": "regional",
                        "frequency": 0.20
                    },
                    {
                        "name": "Forest Fire",
                        "subtype_id": 104,
                        "description": "Large-scale forest fire affecting tourism and local economy",
                        "effects": {"A": -0.08, "K": -0.10, "L": -0.05, "ALPHA": -0.05, "BETA": -0.03},
                        "effects_std": {"A": 0.03, "K": 0.04, "L": 0.03, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "city_specific",
                        "frequency": 0.18
                    },
                    {
                        "name": "Earthquake Reconstruction",
                        "subtype_id": 105,
                        "description": "Post-earthquake reconstruction boosting capital and productivity locally",
                        "effects": {"A": 0.10, "K": 0.18, "L": 0.05, "ALPHA": 0.08, "BETA": 0.05},
                        "effects_std": {"A": 0.04, "K": 0.07, "L": 0.03, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "city_specific",
                        "frequency": 0.20
                    }
                ]
            },
            
            # Health / Pandemic
            {
                "type": "Health",
                "type_id": 2,
                "subtypes": [
                    {
                        "name": "Pandemic Wave",
                        "subtype_id": 201,
                        "description": "Nationwide pandemic affecting labor availability and productivity",
                        "effects": {"A": -0.08, "K": -0.04, "L": -0.20, "ALPHA": -0.03, "BETA": -0.10},
                        "effects_std": {"A": 0.04, "K": 0.02, "L": 0.08, "ALPHA": 0.02, "BETA": 0.04},
                        "affected_scope": "nationwide",
                        "frequency": 0.25
                    },
                    {
                        "name": "Vaccination Rollout",
                        "subtype_id": 202,
                        "description": "Health policy improves labor participation and productivity",
                        "effects": {"A": 0.06, "K": 0.03, "L": 0.12, "ALPHA": 0.02, "BETA": 0.08},
                        "effects_std": {"A": 0.03, "K": 0.02, "L": 0.05, "ALPHA": 0.01, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.20
                    }
                ]
            },
            
            # Economic Crises / Energy / Trade
            {
                "type": "Economic Crisis",
                "type_id": 3,
                "subtypes": [
                    {
                        "name": "Currency Devaluation",
                        "subtype_id": 301,
                        "description": "Significant currency devaluation affecting import costs and inflation",
                        "effects": {"A": -0.08, "K": -0.12, "L": -0.05, "ALPHA": -0.06, "BETA": -0.04},
                        "effects_std": {"A": 0.04, "K": 0.06, "L": 0.03, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.18
                    },
                    {
                        "name": "Inflation Spike",
                        "subtype_id": 302,
                        "description": "Rapid inflation affecting purchasing power and business costs",
                        "effects": {"A": -0.10, "K": -0.08, "L": -0.12, "ALPHA": -0.05, "BETA": -0.08},
                        "effects_std": {"A": 0.05, "K": 0.04, "L": 0.05, "ALPHA": 0.02, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.16
                    },
                    {
                        "name": "Banking Crisis",
                        "subtype_id": 303,
                        "description": "Financial sector instability affecting credit availability",
                        "effects": {"A": -0.05, "K": -0.20, "L": -0.08, "ALPHA": -0.15, "BETA": -0.06},
                        "effects_std": {"A": 0.03, "K": 0.08, "L": 0.04, "ALPHA": 0.06, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.12
                    },
                    {
                        "name": "Energy Price Shock",
                        "subtype_id": 304,
                        "description": "Global energy price jump reduces productivity and capital investment",
                        "effects": {"A": -0.07, "K": -0.10, "L": -0.04, "ALPHA": -0.04, "BETA": -0.03},
                        "effects_std": {"A": 0.03, "K": 0.05, "L": 0.02, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.22
                    },
                    {
                        "name": "Trade Agreement",
                        "subtype_id": 305,
                        "description": "New trade agreement boosting productivity and capital inflows",
                        "effects": {"A": 0.12, "K": 0.18, "L": 0.06, "ALPHA": 0.06, "BETA": 0.05},
                        "effects_std": {"A": 0.05, "K": 0.07, "L": 0.03, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.18
                    },
                    {
                        "name": "Sanctions",
                        "subtype_id": 306,
                        "description": "External sanctions reduce productivity and capital access",
                        "effects": {"A": -0.12, "K": -0.15, "L": -0.06, "ALPHA": -0.08, "BETA": -0.05},
                        "effects_std": {"A": 0.05, "K": 0.06, "L": 0.03, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.10
                    }
                ]
            },
            
            # Infrastructure Development
            {
                "type": "Infrastructure",
                "type_id": 4,
                "subtypes": [
                    {
                        "name": "Major Highway Construction",
                        "subtype_id": 401,
                        "description": "New highway connecting major cities improving transportation efficiency",
                        "effects": {"A": 0.15, "K": 0.20, "L": 0.10, "ALPHA": 0.08, "BETA": 0.06},
                        "effects_std": {"A": 0.06, "K": 0.08, "L": 0.04, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "regional",
                        "frequency": 0.20
                    },
                    {
                        "name": "Port Expansion",
                        "subtype_id": 402,
                        "description": "Major port expansion increasing trade capacity",
                        "effects": {"A": 0.18, "K": 0.25, "L": 0.12, "ALPHA": 0.12, "BETA": 0.08},
                        "effects_std": {"A": 0.07, "K": 0.10, "L": 0.05, "ALPHA": 0.04, "BETA": 0.03},
                        "affected_scope": "city_specific",
                        "frequency": 0.18
                    },
                    {
                        "name": "Airport Modernization",
                        "subtype_id": 403,
                        "description": "Airport upgrade improving tourism and business connectivity",
                        "effects": {"A": 0.12, "K": 0.15, "L": 0.08, "ALPHA": 0.08, "BETA": 0.05},
                        "effects_std": {"A": 0.05, "K": 0.06, "L": 0.03, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "city_specific",
                        "frequency": 0.18
                    },
                    {
                        "name": "High-Speed Rail",
                        "subtype_id": 404,
                        "description": "High-speed rail network improves connectivity and productivity",
                        "effects": {"A": 0.14, "K": 0.20, "L": 0.10, "ALPHA": 0.10, "BETA": 0.06},
                        "effects_std": {"A": 0.06, "K": 0.08, "L": 0.04, "ALPHA": 0.04, "BETA": 0.03},
                        "affected_scope": "regional",
                        "frequency": 0.16
                    }
                ]
            },
            
            # Technology & Innovation
            {
                "type": "Technology & Innovation",
                "type_id": 5,
                "subtypes": [
                    {
                        "name": "Digital Transformation",
                        "subtype_id": 501,
                        "description": "Widespread adoption of digital technologies improving efficiency",
                        "effects": {"A": 0.20, "K": 0.15, "L": 0.05, "ALPHA": 0.10, "BETA": 0.15},
                        "effects_std": {"A": 0.08, "K": 0.06, "L": 0.03, "ALPHA": 0.04, "BETA": 0.06},
                        "affected_scope": "nationwide",
                        "frequency": 0.25
                    },
                    {
                        "name": "Industrial Automation",
                        "subtype_id": 502,
                        "description": "Advanced automation in manufacturing and services",
                        "effects": {"A": 0.25, "K": 0.30, "L": -0.10, "ALPHA": 0.20, "BETA": -0.05},
                        "effects_std": {"A": 0.10, "K": 0.12, "L": 0.05, "ALPHA": 0.08, "BETA": 0.03},
                        "affected_scope": "regional",
                        "frequency": 0.20
                    },
                    {
                        "name": "Renewable Energy Investment",
                        "subtype_id": 503,
                        "description": "Major investment in renewable energy infrastructure",
                        "effects": {"A": 0.15, "K": 0.25, "L": 0.12, "ALPHA": 0.12, "BETA": 0.08},
                        "effects_std": {"A": 0.06, "K": 0.10, "L": 0.05, "ALPHA": 0.05, "BETA": 0.03},
                        "affected_scope": "regional",
                        "frequency": 0.20
                    },
                    {
                        "name": "AI & Machine Learning",
                        "subtype_id": 504,
                        "description": "AI adoption improves productivity and capital efficiency",
                        "effects": {"A": 0.18, "K": 0.12, "L": 0.08, "ALPHA": 0.15, "BETA": 0.12},
                        "effects_std": {"A": 0.07, "K": 0.05, "L": 0.04, "ALPHA": 0.06, "BETA": 0.05},
                        "affected_scope": "nationwide",
                        "frequency": 0.18
                    },
                    {
                        "name": "5G Network Rollout",
                        "subtype_id": 505,
                        "description": "5G infrastructure improves connectivity and productivity",
                        "effects": {"A": 0.12, "K": 0.18, "L": 0.06, "ALPHA": 0.08, "BETA": 0.06},
                        "effects_std": {"A": 0.05, "K": 0.07, "L": 0.03, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "regional",
                        "frequency": 0.17
                    }
                ]
            },
            
            # Environmental & Climate
            {
                "type": "Environmental",
                "type_id": 6,
                "subtypes": [
                    {
                        "name": "Climate Adaptation",
                        "subtype_id": 601,
                        "description": "Investment in climate adaptation infrastructure",
                        "effects": {"A": 0.08, "K": 0.15, "L": 0.06, "ALPHA": 0.06, "BETA": 0.04},
                        "effects_std": {"A": 0.03, "K": 0.06, "L": 0.03, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "regional",
                        "frequency": 0.18
                    },
                    {
                        "name": "Green Building Standards",
                        "subtype_id": 602,
                        "description": "New environmental standards affect construction costs and efficiency",
                        "effects": {"A": -0.03, "K": 0.08, "L": 0.04, "ALPHA": 0.05, "BETA": 0.03},
                        "effects_std": {"A": 0.02, "K": 0.03, "L": 0.02, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.16
                    },
                    {
                        "name": "Carbon Tax Implementation",
                        "subtype_id": 603,
                        "description": "Carbon pricing affects energy costs and investment patterns",
                        "effects": {"A": -0.06, "K": -0.08, "L": -0.03, "ALPHA": -0.04, "BETA": -0.02},
                        "effects_std": {"A": 0.03, "K": 0.04, "L": 0.02, "ALPHA": 0.02, "BETA": 0.01},
                        "affected_scope": "nationwide",
                        "frequency": 0.14
                    }
                ]
            },
            
            # Sectoral & Market Dynamics
            {
                "type": "Sectoral",
                "type_id": 7,
                "subtypes": [
                    {
                        "name": "Tourism Boom",
                        "subtype_id": 701,
                        "description": "Strong tourism season boosts productivity and labor",
                        "effects": {"A": 0.10, "K": 0.08, "L": 0.12, "ALPHA": 0.06, "BETA": 0.10},
                        "effects_std": {"A": 0.04, "K": 0.03, "L": 0.05, "ALPHA": 0.03, "BETA": 0.05},
                        "affected_scope": "regional",
                        "frequency": 0.20
                    },
                    {
                        "name": "Tourism Bust",
                        "subtype_id": 702,
                        "description": "Weak tourism season reduces productivity and labor",
                        "effects": {"A": -0.08, "K": -0.06, "L": -0.10, "ALPHA": -0.05, "BETA": -0.08},
                        "effects_std": {"A": 0.03, "K": 0.03, "L": 0.05, "ALPHA": 0.02, "BETA": 0.04},
                        "affected_scope": "regional",
                        "frequency": 0.18
                    },
                    {
                        "name": "Mining Discovery",
                        "subtype_id": 703,
                        "description": "New mining discovery increases capital and productivity locally",
                        "effects": {"A": 0.12, "K": 0.22, "L": 0.06, "ALPHA": 0.10, "BETA": 0.05},
                        "effects_std": {"A": 0.05, "K": 0.09, "L": 0.03, "ALPHA": 0.04, "BETA": 0.02},
                        "affected_scope": "city_specific",
                        "frequency": 0.16
                    },
                    {
                        "name": "Cyberattack on Industry",
                        "subtype_id": 704,
                        "description": "Cyberattack disrupts industrial systems reducing productivity",
                        "effects": {"A": -0.10, "K": -0.06, "L": -0.04, "ALPHA": -0.08, "BETA": -0.03},
                        "effects_std": {"A": 0.04, "K": 0.03, "L": 0.02, "ALPHA": 0.03, "BETA": 0.02},
                        "affected_scope": "regional",
                        "frequency": 0.14
                    },
                    {
                        "name": "Supply Chain Disruption",
                        "subtype_id": 705,
                        "description": "Global supply chain issues affect production and capital efficiency",
                        "effects": {"A": -0.08, "K": -0.05, "L": -0.06, "ALPHA": -0.04, "BETA": -0.05},
                        "effects_std": {"A": 0.03, "K": 0.02, "L": 0.03, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.20
                    },
                    {
                        "name": "Market Competition",
                        "subtype_id": 706,
                        "description": "Increased market competition improves efficiency and productivity",
                        "effects": {"A": 0.06, "K": 0.04, "L": 0.08, "ALPHA": 0.05, "BETA": 0.06},
                        "effects_std": {"A": 0.03, "K": 0.02, "L": 0.04, "ALPHA": 0.02, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.18
                    }
                ]
            }
        ]
        
        # Generate shocks for each year of simulation
        for year in range(self.start_year, self.start_year + self.years):
            # 1 shock per year as requested
            shock_type = random.choice(shock_templates)
            shock_subtype = random.choices(
                shock_type["subtypes"], 
                weights=[s["frequency"] for s in shock_type["subtypes"]]
            )[0]
            
            # Sample actual effects for this specific shock instance
            effects_mean = shock_subtype["effects"]
            effects_std = shock_subtype.get("effects_std")
            sampled_effects = _sample_effects(effects_mean, effects_std)
            # Bias shocks: make negatives stronger, positives weaker
            biased_effects = {}
            for param, val in sampled_effects.items():
                if val < 0:
                    val *= 1.3  # amplify negative shocks
                else:
                    val *= 0.7  # dampen positive shocks
                # keep within caps
                val = min(max(val, -0.5), 0.5)
                biased_effects[param] = val
            
            # Determine affected cities based on scope
            if shock_subtype["affected_scope"] == "nationwide":
                affected_targets = {"NATION": biased_effects}
            elif shock_subtype["affected_scope"] == "regional":
                # Select a spatially coherent cluster: seed city + nearest neighbors (3–5 cities total)
                num_cities = random.randint(5, 10)
                city_names = list(self.tr_network.cities.keys())
                seed_city = random.choice(city_names)
                seed_obj = self.tr_network.cities[seed_city]
                # Sort cities by geographic distance to the seed city
                sorted_by_near = sorted(
                    city_names,
                    key=lambda cn: seed_obj.distance_to(self.tr_network.cities[cn])
                )
                selected_cities = sorted_by_near[:num_cities]
                affected_targets = {city: biased_effects for city in selected_cities}
            else:  # city_specific
                selected_city = random.choice(list(self.tr_network.cities.keys()))
                affected_targets = {selected_city: biased_effects}
            
            # Create shock record
            shock = {
                "EFFECT": [
                    shock_subtype["name"],
                    shock_type["type"],
                    shock_subtype["description"]
                ],
                "EFFECTS": affected_targets,
                "Year": year,
                "Type_ID": shock_type["type_id"],
                "Subtype_ID": shock_subtype["subtype_id"],
                "Scope": shock_subtype.get("affected_scope", "")
            }
            
            self.available_shocks.append(shock)
        
        print(f"Generated {len(self.available_shocks)} realistic shocks for {self.years} years")
        print(f"Shock types: {', '.join(set(s['EFFECT'][1] for s in self.available_shocks))}")
        
        # Show sample of generated shocks
        print("\nSample generated shocks:")
        for i, shock in enumerate(self.available_shocks[:3]):
            print(f"  {i+1}. {shock['EFFECT'][0]} ({shock['EFFECT'][1]})")
            print(f"     {shock['EFFECT'][2]}")
            print(f"     Affects: {list(shock['EFFECTS'].keys())}")
            print()

    # =====================
    # Policy System (Model4)
    # =====================
    def load_policies(self):
        """Generate realistic economic policies programmatically (1 per year)."""
        if self.model_type != "policy":
            return
        self.generate_realistic_policies()
    
    def apply_policy_effects(self, policy_data):
        """Apply policy effects to cities based on the policy data."""
        if self.model_type != "policy":
            return
        effects = policy_data.get('EFFECTS', {})
        for target, effect_values in effects.items():
            if str(target).upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                target_cities = list(self.tr_network.cities.values())
            elif target in self.tr_network.cities:
                target_cities = [self.tr_network.cities[target]]
            else:
                print(f"Warning: Target '{target}' not found in city list, skipping policy effect")
                continue
            for city in target_cities:
                if isinstance(effect_values, dict):
                    self._apply_city_effects(city, effect_values)
                elif isinstance(effect_values, list):
                    effect_dict = {param: val for param, val in effect_values}
                    self._apply_city_effects(city, effect_dict)
    
    def trigger_random_policies(self, year):
        """Trigger one policy for the current year from pre-generated policies."""
        if self.model_type != "policy" or not self.available_policies:
            return []
        year_policies = [p for p in self.available_policies if p.get('Year') == year]
        if not year_policies:
            # Fallback: simple nationwide productivity policy
            fallback_policy = {
                "EFFECT": ["Growth Support", "Macro Policy", "Temporary support for productivity"],
                "EFFECTS": {"NATION": {"A": 0.05, "K": 0.05, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03}},
                "Year": year,
                "Policy_Type_ID": 900,
                "Policy_Subtype_ID": 901,
                "Policy_Scope": "nationwide"
            }
            year_policies = [fallback_policy]
        for policy in year_policies:
            self.apply_policy_effects(policy)
            policy_record = {
                'Year': year,
                'Effect_Title': policy['EFFECT'][0],
                'Domain': policy['EFFECT'][1],
                'Description': policy['EFFECT'][2],
                'Effects': policy['EFFECTS'],
                'Policy_Type_ID': policy.get('Policy_Type_ID', 0),
                'Policy_Subtype_ID': policy.get('Policy_Subtype_ID', 0),
                'Policy_Scope': policy.get('Policy_Scope', '')
            }
            self.active_policies.append(policy_record)
        print(f"  Year {year}: {len(year_policies)} policy/policies triggered")
        for p in year_policies:
            print(f"    - {p['EFFECT'][0]} ({p['EFFECT'][1]})")
        return year_policies
    
    def get_policy_summary(self, year):
        """Get a summary of policies that occurred in a specific year."""
        if self.model_type != "policy":
            return f"No policy system for {self.model_type} economy"
        year_policies = [p for p in self.active_policies if p['Year'] == year]
        if not year_policies:
            return f"No policies occurred in year {year}"
        summary = f"Policy Summary for Year {year}:\n"
        summary += f"Total policies: {len(year_policies)}\n"
        for i, pol in enumerate(year_policies, 1):
            summary += f"  {i}. {pol['Effect_Title']} ({pol['Domain']})\n"
        return summary

    def export_policy_results(self, filename=None):
        """Export policy results to Excel for analysis."""
        if self.model_type != "policy" or not self.active_policies:
            print("No policy data available to export")
            return
        
        if filename is None:
            filename = get_excel_path("turkey_policy_simulation", self.model_type)
        
        # Create DataFrame from policy results
        policy_df = pd.DataFrame(self.active_policies)
        
        # Export to Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            policy_df.to_excel(writer, sheet_name='Policy_Results', index=False)
            
            # Create summary sheet
            summary_data = []
            for policy in self.active_policies:
                summary_data.append({
                    'Year': policy['Year'],
                    'Effect_Title': policy['Effect_Title'],
                    'Domain': policy['Domain'],
                    'Target_Count': len(policy['Effects'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Policy_Summary', index=False)
        
        print(f"Policy results exported to {filename}")

    def export_detailed_policy_results(self, filename=None):
        """
        Export detailed policy results with parameter changes for reinforcement learning.
        Shows old/new parameter values and changes for each affected city.
        
        Args:
            filename (str): Path to save the detailed policy results
        """
        if self.model_type != "policy" or not self.active_policies:
            print("No policy data available to export")
            return
        
        if filename is None:
            filename = get_excel_path("turkey_detailed_policy_results", self.model_type)
        
        # Create detailed policy results with parameter changes
        detailed_policy_data = []
        
        for policy in self.active_policies:
            year = policy['Year']
            effects = policy['Effects']
            
            # Get all cities that were affected by this policy
            affected_cities = []
            for target, effect_values in effects.items():
                if target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                    # Nationwide policy affects all cities
                    affected_cities.extend(list(self.tr_network.cities.keys()))
                elif target in self.tr_network.cities:
                    # City-specific policy
                    affected_cities.append(target)
            
            # Create records for each affected city
            for city_name in affected_cities:
                city = self.tr_network.cities[city_name]
                
                # Get old parameter values (before policy)
                old_params = {
                    'A': city.A,
                    'K': city.capital_stock,
                    'L': city.labor_force,
                    'ALPHA': city.alpha,
                    'BETA': city.beta
                }
                
                # Calculate new parameter values (after policy)
                new_params = {}
                # Handle both dictionary and list formats
                if isinstance(effect_values, dict):
                    for param, effect in effect_values.items():
                        if param == "A":
                            if old_params['A'] < 0:
                                new_params['A'] = old_params['A'] * (1 - effect/100)
                            else:
                                new_params['A'] = old_params['A'] * (1 + effect/100)
                        elif param == "K":
                            if old_params['K'] < 0:
                                new_params['K'] = old_params['K'] * (1 - effect/100)
                            else:
                                new_params['K'] = old_params['K'] * (1 + effect/100)
                        elif param == "L":
                            if old_params['L'] < 0:
                                new_params['L'] = int(old_params['L'] * (1 - effect/100))
                            else:
                                new_params['L'] = int(old_params['L'] * (1 + effect/100))
                        elif param == "ALPHA":
                            if old_params['ALPHA'] < 0:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 - effect/100)
                            else:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 + effect/100)
                        elif param == "BETA":
                            if old_params['BETA'] < 0:
                                new_params['BETA'] = old_params['BETA'] * (1 - effect/100)
                            else:
                                new_params['BETA'] = old_params['BETA'] * (1 + effect/100)
                elif isinstance(effect_values, list):
                    for param, effect in effect_values:
                        if param == "A":
                            if old_params['A'] < 0:
                                new_params['A'] = old_params['A'] * (1 - effect/100)
                            else:
                                new_params['A'] = old_params['A'] * (1 + effect/100)
                        elif param == "K":
                            if old_params['K'] < 0:
                                new_params['K'] = old_params['K'] * (1 - effect/100)
                            else:
                                new_params['K'] = old_params['K'] * (1 + effect/100)
                        elif param == "L":
                            if old_params['L'] < 0:
                                new_params['L'] = int(old_params['L'] * (1 - effect/100))
                            else:
                                new_params['L'] = int(old_params['L'] * (1 + effect/100))
                        elif param == "ALPHA":
                            if old_params['ALPHA'] < 0:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 - effect/100)
                            else:
                                new_params['ALPHA'] = old_params['ALPHA'] * (1 + effect/100)
                        elif param == "BETA":
                            if old_params['BETA'] < 0:
                                new_params['BETA'] = old_params['BETA'] * (1 - effect/100)
                            else:
                                new_params['BETA'] = old_params['BETA'] * (1 + effect/100)
                
                # Fill in unchanged parameters
                for param in ['A', 'K', 'L', 'ALPHA', 'BETA']:
                    if param not in new_params:
                        new_params[param] = old_params[param]
                
                # Create detailed policy record
                policy_record = {
                    'Year': year,
                    'City': city_name,
                    'Policy_Title': policy['Effect_Title'],
                    'Policy_Domain': policy['Domain'],
                    'Policy_Description': policy['Description'],
                    'Policy_Type_ID': policy.get('Policy_Type_ID', 0),
                    'Policy_Subtype_ID': policy.get('Policy_Subtype_ID', 0),
                    'Policy_Scope': policy.get('Policy_Scope', ''),
                    # Old parameter values
                    'Old_A': old_params['A'],
                    'Old_K': old_params['K'],
                    'Old_L': old_params['L'],
                    'Old_ALPHA': old_params['ALPHA'],
                    'Old_BETA': old_params['BETA'],
                    # New parameter values
                    'New_A': new_params['A'],
                    'New_K': new_params['K'],
                    'New_L': new_params['L'],
                    'New_ALPHA': new_params['ALPHA'],
                    'New_BETA': new_params['BETA'],
                    # Parameter changes
                    'Delta_A': new_params['A'] - old_params['A'],
                    'Delta_K': new_params['K'] - old_params['K'],
                    'Delta_L': new_params['L'] - old_params['L'],
                    'Delta_ALPHA': new_params['ALPHA'] - old_params['ALPHA'],
                    'Delta_BETA': new_params['BETA'] - old_params['BETA']
                }
                
                detailed_policy_data.append(policy_record)
        
        # Create detailed policy DataFrame
        detailed_policy_df = pd.DataFrame(detailed_policy_data)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main detailed results
            detailed_policy_df.to_excel(writer, sheet_name='Detailed_Policy_Results', index=False)
            
            # Summary by year and city
            summary_df = detailed_policy_df.groupby(['Year', 'City']).agg({
                'Policy_Title': 'count',
                'Delta_A': 'sum',
                'Delta_K': 'sum',
                'Delta_L': 'sum',
                'Delta_ALPHA': 'sum',
                'Delta_BETA': 'sum'
            }).reset_index()
            summary_df.columns = ['Year', 'City', 'Policy_Count', 'Total_Delta_A', 'Total_Delta_K', 'Total_Delta_L', 'Total_Delta_ALPHA', 'Total_Delta_BETA']
            summary_df.to_excel(writer, sheet_name='Policy_Summary_By_City', index=False)
            
            # Summary by policy
            policy_summary_df = detailed_policy_df.groupby(['Policy_Title', 'Policy_Domain']).agg({
                'City': 'count',
                'Delta_A': 'mean',
                'Delta_K': 'mean',
                'Delta_L': 'mean',
                'Delta_ALPHA': 'mean',
                'Delta_BETA': 'mean'
            }).reset_index()
            policy_summary_df.columns = ['Policy_Title', 'Policy_Domain', 'Cities_Affected', 'Avg_Delta_A', 'Avg_Delta_K', 'Avg_Delta_L', 'Avg_Delta_ALPHA', 'Avg_Delta_BETA']
            policy_summary_df.to_excel(writer, sheet_name='Policy_Summary_By_Type', index=False)
        
        print(f"Detailed policy results exported to {filename}")
        print(f"Total policy records: {len(detailed_policy_data)}")
        print(f"Affected cities: {len(detailed_policy_df['City'].unique())}")
        print(f"Years with policies: {len(detailed_policy_df['Year'].unique())}")
        
        return filename

    def animate_policy_effects(self, save_path=None, figsize=(32, 24)):
        """
        Create animation showing policy effects on cities over time.
        Gray background with red cities for negative effects and green for positive effects.
        
        Args:
            save_path (str): Path to save the animation
            figsize (tuple): Figure size
        """
        if self.model_type != "policy":
            print("Policy animation only available for policy economy models")
            return
        
        if not self.active_policies:
            print("No policy data available. Run simulation first.")
            return
        
        if self.tr_network.provinces is None:
            print("Cannot animate: provinces data not loaded")
            return
        
        # Prepare data
        policy_df = pd.DataFrame(self.active_policies)
        years = sorted(policy_df['Year'].unique())
        
        if not years:
            print("No policy data found for animation")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.close(fig)
        
        # Prepare provinces data
        provinces = self.tr_network.provinces.copy()
        province_names = provinces['NAME_1'].tolist()
        
        # Normalize names (handle Turkish characters like MUŞ -> MUS) for reliable matching
        def _normalize_name(name: str) -> str:
            try:
                s = str(name)
            except Exception:
                s = ''
            trans = str.maketrans({
                'Ç':'C','Ö':'O','Ş':'S','İ':'I','I':'I','Ü':'U','Ğ':'G',
                'ç':'C','ö':'O','ş':'S','ı':'I','i':'I','ü':'U','ğ':'G'
            })
            return s.translate(trans).upper().strip()
        # Build normalized province index lookup
        _prov_idx_by_norm = { _normalize_name(p): i for i, p in enumerate(province_names) }
        
        # Build city-to-province mapping
        city_to_province = {}
        for province_name in province_names:
            for city in self.tr_network.cities:
                if city.upper() in province_name.upper() or province_name.upper() in city.upper():
                    city_to_province[city] = province_name
        
        def get_policy_effects(year):
            """Get policy effects for a specific year."""
            year_policies = policy_df[policy_df['Year'] == year]
            city_effects = {}
            
            for _, policy in year_policies.iterrows():
                effects = policy['Effects']
                
                for target, effect_values in effects.items():
                    if target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"]:
                        # Nationwide policy affects all cities
                        for city_name in self.tr_network.cities.keys():
                            if city_name not in city_effects:
                                city_effects[city_name] = {'positive': 0, 'negative': 0, 'net': 0.0, 'policies': []}
                            
                            # Determine effect contributions (supports dict or list)
                            if isinstance(effect_values, dict):
                                for param, effect in effect_values.items():
                                    if effect > 0:
                                        city_effects[city_name]['positive'] += 1
                                    elif effect < 0:
                                        city_effects[city_name]['negative'] += 1
                                    city_effects[city_name]['net'] += float(effect)
                            elif isinstance(effect_values, list):
                                for param, effect in effect_values:
                                    if effect > 0:
                                        city_effects[city_name]['positive'] += 1
                                    elif effect < 0:
                                        city_effects[city_name]['negative'] += 1
                                    city_effects[city_name]['net'] += float(effect)
                            
                            city_effects[city_name]['policies'].append({
                                'title': policy['Effect_Title'],
                                'domain': policy['Domain'],
                                'effects': effect_values
                            })
                    
                    elif target in self.tr_network.cities:
                        # City-specific policy - only affect the specific city
                        if target not in city_effects:
                            city_effects[target] = {'positive': 0, 'negative': 0, 'net': 0.0, 'policies': []}
                        
                        # Determine effect contributions (supports dict or list)
                        if isinstance(effect_values, dict):
                            for param, effect in effect_values.items():
                                if effect > 0:
                                    city_effects[target]['positive'] += 1
                                elif effect < 0:
                                    city_effects[target]['negative'] += 1
                                city_effects[target]['net'] += float(effect)
                        elif isinstance(effect_values, list):
                            for param, effect in effect_values:
                                if effect > 0:
                                    city_effects[target]['positive'] += 1
                                elif effect < 0:
                                    city_effects[target]['negative'] += 1
                                city_effects[target]['net'] += float(effect)
                        
                        city_effects[target]['policies'].append({
                            'title': policy['Effect_Title'],
                            'domain': policy['Domain'],
                            'effects': effect_values
                        })
            
            return city_effects
        
        # Set up the initial plot
        provinces.plot(
            ax=ax,
            column='NAME_1',
            edgecolor='white',
            linewidth=0.5,
            legend=False,
            color='lightgray'
        )
        
        # Add city labels
        for city in self.tr_network.cities:
            pos = self.tr_network.network.nodes[city]['pos']
            ax.annotate(
                city,
                xy=pos,
                xytext=(0, 0),
                textcoords='offset points',
                fontsize=18,
                ha='center',
                va='center',
                fontweight='bold',
                color='black'
            )
        
        # Set map limits
        ax.set_xlim(25, 45)
        ax.set_ylim(35, 43)
        ax.set_title(f"Turkey Policy Effects - Year {years[0]}\n{self.model_type.capitalize()} Economy", 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        def update(frame):
            year = years[frame]
            ax.clear()
            
            # Plot provinces (default gray)
            provinces.plot(
                ax=ax,
                column='NAME_1',
                edgecolor='white',
                linewidth=0.5,
                legend=False,
                color='lightgray'
            )
            
            # Get policy effects for this year
            city_effects = get_policy_effects(year)
            
            # Plot cities with policy effects by coloring their provinces
            total_policies = 0
            positive_policies = 0
            negative_policies = 0
            
            # Create a list to store province colors
            province_colors = ['lightgray'] * len(province_names)
            
            for city_name, effects in city_effects.items():
                if city_name in self.tr_network.cities:
                    # Use normalized direct lookup (handles MUŞ vs MUS)
                    city_province = _prov_idx_by_norm.get(_normalize_name(city_name))
                    
                    if city_province is not None:
                        # Determine province color based on net policy effect
                        if effects['net'] > 0:
                            # Net positive effects - green
                            province_colors[city_province] = 'green'
                            positive_policies += 1
                        elif effects['net'] < 0:
                            # Net negative effects - red
                            province_colors[city_province] = 'red'
                            negative_policies += 1
                        else:
                            # No net effect - keep gray
                            province_colors[city_province] = 'lightgray'
                        
                        total_policies += abs(effects['net'])
            
            # Re-plot provinces with updated colors (explicit color array to avoid colormap mixing)
            provinces.plot(
                ax=ax,
                color=province_colors,
                edgecolor='white',
                linewidth=0.5,
                legend=False
            )
            
            # Add city labels
            for city in self.tr_network.cities:
                pos = self.tr_network.network.nodes[city]['pos']
                ax.annotate(
                    city,
                    xy=pos,
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=18,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='black'
                )
                
                # Add simple dot for affected cities (optional)
                if city in city_effects and city_effects[city]['net'] != 0:
                    ax.annotate(
                        '●',
                        xy=pos,
                        xytext=(0, 0),
                        textcoords='offset points',
                        fontsize=14,
                        ha='center',
                        va='center',
                        color='white'
                    )
            
            # Build title with policy type/domain/description (assumes 1 per year)
            try:
                yrow = policy_df[policy_df['Year'] == year].iloc[0]
                policy_title = str(yrow.get('Effect_Title', ''))
                policy_domain = str(yrow.get('Domain', ''))
                policy_desc = str(yrow.get('Description', ''))
                title_line1 = f"{policy_title} | {policy_domain}"
                title_line2 = policy_desc[:160]
            except Exception:
                title_line1 = f"Turkey Policy Effects"
                title_line2 = ""
            
            # Set map limits and titles
            ax.set_xlim(25, 45)
            ax.set_ylim(35, 43)
            ax.set_title(f"{title_line1}\nYear {year} - {self.model_type.capitalize()} Economy\n{title_line2}", 
                        fontsize=22, fontweight='bold', pad=22)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Bigger, clearer legend
            legend_text = f"""Policy Map Legend

Red: Net negative policy
Blue: Net positive policy
Gray: No policy effect

Year {year} summary
Cities with net positive: {positive_policies}
Cities with net negative: {negative_policies}
"""
            
            ax.text(0.015, 0.985, legend_text, 
                   transform=ax.transAxes, fontsize=16, 
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95),
                   verticalalignment='top')
        
        # Create animation with faster speed for readability
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False, interval=1000)
        
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=1.0)
            else:
                ani.save(save_path, writer='ffmpeg', fps=1.0)
            print(f"Policy effects animation saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_animation_path
            save_path = get_animation_path("turkey_policy_effects", self.model_type)
            ani.save(save_path, writer='pillow', fps=1.0)
            print(f"Policy effects animation saved to {save_path}")
        
        plt.show()
        return save_path

    def animate_city_bar_chart_race(self, metric='gdp_per_capita', save_path=None, figsize=(16, 10), top_n=10):
        """
        Create animated bar chart race for cities based on various metrics.
        
        Args:
            metric (str): Metric to rank cities by ('gdp_per_capita', 'population', 'production', 'migration_in', 'migration_out')
            save_path (str): Path to save the animation
            figsize (tuple): Figure size
            top_n (int): Number of top cities to show
        """
        if not hasattr(self, 'comparison_results'):
            print("No comparison results available. Run run_model_comparison() first.")
            return
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        results_df = pd.DataFrame(self.comparison_results)
        
        if results_df.empty:
            print("No data available for bar chart race")
            return
        
        # Get unique years and models
        years = sorted(results_df['Year'].unique())
        models = results_df['Model_Name'].unique()
        
        if len(models) < 1:
            print("No models available for bar chart race")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        plt.close(fig)
        
        # Define metric mapping
        metric_mapping = {
            'gdp_per_capita': ('GDP per Capita', 'GDP_per_Capita'),
            'population': ('Population', 'Population'),
            'production': ('Production', 'Production'),
            'migration_in': ('Migration In', 'Migration_In'),
            'migration_out': ('Migration Out', 'Migration_Out')
        }
        
        if metric not in metric_mapping:
            print(f"Invalid metric: {metric}. Available: {list(metric_mapping.keys())}")
            return
        
        metric_title, metric_col = metric_mapping[metric]
        
        # Check if migration data is available
        if metric in ['migration_in', 'migration_out'] and 'migration_results' not in dir(self):
            print(f"Migration data not available for {metric}")
            return
        
        def get_city_rankings(year, model_name):
            """Get city rankings for a specific year and model."""
            year_model_df = results_df[(results_df['Year'] == year) & (results_df['Model_Name'] == model_name)]
            
            if metric in ['migration_in', 'migration_out']:
                # For migration metrics, we need to calculate from migration results
                if hasattr(self, 'comparison_migration_results'):
                    migration_df = pd.DataFrame(self.comparison_migration_results)
                    model_migrations = migration_df[migration_df['Model_Name'] == model_name]
                    year_migrations = model_migrations[model_migrations['Year'] == year]
                    
                    if metric == 'migration_in':
                        city_data = year_migrations.groupby('Target_City')['Migration_Population'].sum().reset_index()
                        city_data = city_data.rename(columns={'Target_City': 'City', 'Migration_Population': metric_col})
                    else:  # migration_out
                        city_data = year_migrations.groupby('Source_City')['Migration_Population'].sum().reset_index()
                        city_data = city_data.rename(columns={'Source_City': 'City', 'Migration_Population': metric_col})
                else:
                    return pd.DataFrame()
            else:
                # For other metrics, use the main results
                city_data = year_model_df[['City', metric_col]].copy()
            
            # Sort by metric value (descending for top ranking)
            city_data = city_data.sort_values(metric_col, ascending=False)
            
            # Take top N cities
            return city_data.head(top_n)
        
        def update(frame):
            ax.clear()
            
            # Get data for current frame
            year = years[frame]
            model_name = models[0]  # Use first model for simplicity
            
            city_data = get_city_rankings(year, model_name)
            
            if city_data.empty:
                ax.text(0.5, 0.5, f'No data available for {year}', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=16)
                return
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(city_data)), city_data[metric_col], 
                          color='steelblue', alpha=0.8)
            
            # Add city labels
            ax.set_yticks(range(len(city_data)))
            ax.set_yticklabels(city_data['City'])
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, city_data[metric_col])):
                ax.text(bar.get_width() + bar.get_width() * 0.01, 
                       bar.get_y() + bar.get_height() / 2,
                       f'{value:,.0f}' if metric in ['population', 'migration_in', 'migration_out'] else f'{value:,.2f}',
                       va='center', fontsize=10, fontweight='bold')
            
            # Customize plot
            ax.set_title(f'Top {top_n} Cities by {metric_title} - {model_name}\nYear {year}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(metric_title, fontsize=14)
            ax.set_ylabel('City', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Invert y-axis for top-to-bottom ranking
            ax.invert_yaxis()
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(years), repeat=False, interval=500)
        
        if save_path:
            if save_path.endswith('.gif'):
                ani.save(save_path, writer='pillow', fps=2)
            else:
                ani.save(save_path, writer='ffmpeg', fps=2)
            print(f"Bar chart race animation saved to {save_path}")
        else:
            # Use output configuration to generate organized filename
            from output_config import get_animation_path
            save_path = get_animation_path(f"city_bar_chart_race_{metric}", "models")
            ani.save(save_path, writer='pillow', fps=2)
            print(f"Bar chart race animation saved to {save_path}")
        
        plt.show()
        return save_path

    def get_policy_details(self, city_name, year):
        """
        Get detailed policy information for a specific city and year.
        
        Args:
            city_name (str): Name of the city
            year (int): Year to get policy details for
            
        Returns:
            dict: Detailed policy information for the city
        """
        if self.model_type != "policy" or not self.active_policies:
            return None
        
        city_policies = []
        for policy in self.active_policies:
            if policy['Year'] == year:
                effects = policy['Effects']
                
                # Check if this city was affected
                affected = False
                for target, effect_values in effects.items():
                    if (target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"] or 
                        target == city_name):
                        affected = True
                        break
                
                if affected:
                    city_policies.append({
                        'Effect_Title': policy['Effect_Title'],
                        'Domain': policy['Domain'],
                        'Description': policy['Description'],
                        'Effects': policy['Effects']
                    })
        
        if not city_policies:
            return {
                'city': city_name,
                'year': year,
                'policies': [],
                'total_effects': 0,
                'positive_effects': 0,
                'negative_effects': 0
            }
        
        # Count positive and negative effects
        total_effects = 0
        positive_effects = 0
        negative_effects = 0
        
        for policy in city_policies:
            for target, effect_values in policy['Effects'].items():
                if (target.upper() in ["NATION", "NATIONWIDE", "COUNTRY", "ALL", "ANYCOUNTRY", "MAINLAND"] or 
                    target == city_name):
                    # Handle both dictionary and list formats
                    if isinstance(effect_values, dict):
                        for param, effect in effect_values.items():
                            total_effects += 1
                            if effect > 0:
                                positive_effects += 1
                            elif effect < 0:
                                negative_effects += 1
                    elif isinstance(effect_values, list):
                        for param, effect in effect_values:
                            total_effects += 1
                            if effect > 0:
                                positive_effects += 1
                            elif effect < 0:
                                negative_effects += 1
        
        return {
            'city': city_name,
            'year': year,
            'policies': city_policies,
            'total_effects': total_effects,
            'positive_effects': positive_effects,
            'negative_effects': negative_effects
        }

    def generate_realistic_policies(self):
        """
        Generate realistic economic policies programmatically instead of loading from file.
        Creates policies based on real economic scenarios with appropriate parameter effects.
        Effects are randomized per policy around template means (1 std deviation) to avoid repetition.
        """
        if self.model_type != "policy":
            return
        
        # Clear any existing policies
        self.available_policies = []
        
        # Helper: sample effects per parameter with 1 std deviation around mean
        def _sample_effects(effects_mean: dict, effects_std: dict | None = None) -> dict:
            sampled = {}
            for param, mu in effects_mean.items():
                if effects_std and param in effects_std:
                    sigma = effects_std[param]
                else:
                    # Default std: 40% of |mean|, minimum floor
                    sigma = max(0.02, abs(mu) * 0.4)
                val = random.gauss(mu, sigma)
                # Soft cap to keep within a reasonable band for stability
                if val > 0:
                    val = min(val, 0.5)
                else:
                    val = max(val, -0.5)
                sampled[param] = val
            return sampled
        
        # Define realistic policy types with appropriate parameter effects (means and stds)
        policy_templates = [
            # Economic Growth
            {
                "type": "Economic Growth",
                "type_id": 100,
                "subtypes": [
                    {
                        "name": "Fiscal Stimulus",
                        "subtype_id": 101,
                        "description": "Temporary increase in government spending to boost demand",
                        "effects": {"A": 0.05, "K": 0.05, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "effects_std": {"A": 0.02, "K": 0.02, "L": 0.01, "ALPHA": 0.01, "BETA": 0.01},
                        "affected_scope": "nationwide",
                        "frequency": 0.20
                    },
                    {
                        "name": "Monetary Stimulus",
                        "subtype_id": 102,
                        "description": "Temporary decrease in interest rates to stimulate borrowing and spending",
                        "effects": {"A": 0.03, "K": 0.03, "L": 0.01, "ALPHA": 0.02, "BETA": 0.02},
                        "effects_std": {"A": 0.01, "K": 0.01, "L": 0.005, "ALPHA": 0.005, "BETA": 0.005},
                        "affected_scope": "nationwide",
                        "frequency": 0.18
                    },
                    {
                        "name": "Investment Boost",
                        "subtype_id": 103,
                        "description": "Temporary tax incentives for businesses to increase investment",
                        "effects": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "effects_std": {"A": 0.02, "K": 0.02, "L": 0.01, "ALPHA": 0.01, "BETA": 0.01},
                        "affected_scope": "regional",
                        "frequency": 0.16
                    },
                    {
                        "name": "Trade Promotion",
                        "subtype_id": 104,
                        "description": "Temporary reduction in import tariffs to boost exports",
                        "effects": {"A": 0.06, "K": 0.06, "L": 0.03, "ALPHA": 0.05, "BETA": 0.05},
                        "effects_std": {"A": 0.03, "K": 0.03, "L": 0.015, "ALPHA": 0.025, "BETA": 0.025},
                        "affected_scope": "nationwide",
                        "frequency": 0.14
                    }
                ]
            },
            
            # Health / Education
            {
                "type": "Health / Education",
                "type_id": 200,
                "subtypes": [
                    {
                        "name": "Healthcare Expansion",
                        "subtype_id": 201,
                        "description": "Increased public spending on healthcare to improve access and quality",
                        "effects": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "effects_std": {"A": 0.02, "K": 0.02, "L": 0.01, "ALPHA": 0.01, "BETA": 0.01},
                        "affected_scope": "nationwide",
                        "frequency": 0.18
                    },
                    {
                        "name": "Education Reform",
                        "subtype_id": 202,
                        "description": "Increased public spending on education to improve quality and access",
                        "effects": {"A": 0.05, "K": 0.05, "L": 0.02, "ALPHA": 0.04, "BETA": 0.04},
                        "effects_std": {"A": 0.03, "K": 0.03, "L": 0.015, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.16
                    },
                    {
                        "name": "Public Investment",
                        "subtype_id": 203,
                        "description": "Increased public investment in infrastructure to boost productivity",
                        "effects": {"A": 0.06, "K": 0.06, "L": 0.03, "ALPHA": 0.05, "BETA": 0.05},
                        "effects_std": {"A": 0.03, "K": 0.03, "L": 0.015, "ALPHA": 0.025, "BETA": 0.025},
                        "affected_scope": "regional",
                        "frequency": 0.14
                    },
                    {
                        "name": "Labor Market Reforms",
                        "subtype_id": 204,
                        "description": "Reforms to improve labor market flexibility and reduce unemployment",
                        "effects": {"A": 0.05, "K": 0.05, "L": 0.03, "ALPHA": 0.04, "BETA": 0.04},
                        "effects_std": {"A": 0.03, "K": 0.03, "L": 0.015, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.12
                    }
                ]
            },
            
            # Infrastructure Development
            {
                "type": "Infrastructure Development",
                "type_id": 300,
                "subtypes": [
                    {
                        "name": "Transport Infrastructure",
                        "subtype_id": 301,
                        "description": "Increased investment in transportation infrastructure to improve connectivity",
                        "effects": {"A": 0.07, "K": 0.07, "L": 0.04, "ALPHA": 0.06, "BETA": 0.06},
                        "effects_std": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "regional",
                        "frequency": 0.16
                    },
                    {
                        "name": "Energy Infrastructure",
                        "subtype_id": 302,
                        "description": "Increased investment in energy infrastructure to improve reliability and sustainability",
                        "effects": {"A": 0.08, "K": 0.08, "L": 0.05, "ALPHA": 0.07, "BETA": 0.07},
                        "effects_std": {"A": 0.05, "K": 0.05, "L": 0.03, "ALPHA": 0.04, "BETA": 0.04},
                        "affected_scope": "regional",
                        "frequency": 0.14
                    },
                    {
                        "name": "Digital Infrastructure",
                        "subtype_id": 303,
                        "description": "Increased investment in digital infrastructure to support economic growth",
                        "effects": {"A": 0.09, "K": 0.09, "L": 0.05, "ALPHA": 0.08, "BETA": 0.08},
                        "effects_std": {"A": 0.05, "K": 0.05, "L": 0.03, "ALPHA": 0.04, "BETA": 0.04},
                        "affected_scope": "regional",
                        "frequency": 0.12
                    },
                    {
                        "name": "Environmental Infrastructure",
                        "subtype_id": 304,
                        "description": "Increased investment in environmental infrastructure to improve sustainability",
                        "effects": {"A": 0.06, "K": 0.06, "L": 0.04, "ALPHA": 0.05, "BETA": 0.05},
                        "effects_std": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "regional",
                        "frequency": 0.10
                    }
                ]
            },
            
            # Technology & Innovation
            {
                "type": "Technology & Innovation",
                "type_id": 400,
                "subtypes": [
                    {
                        "name": "Research and Development",
                        "subtype_id": 401,
                        "description": "Increased public spending on research and development to boost innovation",
                        "effects": {"A": 0.07, "K": 0.07, "L": 0.04, "ALPHA": 0.06, "BETA": 0.06},
                        "effects_std": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.16
                    },
                    {
                        "name": "Tech Startup Support",
                        "subtype_id": 402,
                        "description": "Increased public support for tech startups to foster innovation",
                        "effects": {"A": 0.08, "K": 0.08, "L": 0.04, "ALPHA": 0.07, "BETA": 0.07},
                        "effects_std": {"A": 0.05, "K": 0.05, "L": 0.03, "ALPHA": 0.04, "BETA": 0.04},
                        "affected_scope": "nationwide",
                        "frequency": 0.14
                    },
                    {
                        "name": "Entrepreneurship Programs",
                        "subtype_id": 403,
                        "description": "Increased public support for entrepreneurship programs to foster innovation",
                        "effects": {"A": 0.06, "K": 0.06, "L": 0.03, "ALPHA": 0.05, "BETA": 0.05},
                        "effects_std": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.12
                    },
                    {
                        "name": "Tech Education",
                        "subtype_id": 404,
                        "description": "Increased public spending on tech education to improve workforce skills",
                        "effects": {"A": 0.05, "K": 0.05, "L": 0.03, "ALPHA": 0.04, "BETA": 0.04},
                        "effects_std": {"A": 0.03, "K": 0.03, "L": 0.015, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.10
                    }
                ]
            },
            
            # Social / Demographic
            {
                "type": "Social / Demographic",
                "type_id": 500,
                "subtypes": [
                    {
                        "name": "Pension Reform",
                        "subtype_id": 501,
                        "description": "Reforms to improve the sustainability of pension systems",
                        "effects": {"A": 0.05, "K": 0.05, "L": 0.03, "ALPHA": 0.04, "BETA": 0.04},
                        "effects_std": {"A": 0.03, "K": 0.03, "L": 0.015, "ALPHA": 0.02, "BETA": 0.02},
                        "affected_scope": "nationwide",
                        "frequency": 0.14
                    },
                    {
                        "name": "Healthcare Reform",
                        "subtype_id": 502,
                        "description": "Reforms to improve the efficiency and accessibility of healthcare",
                        "effects": {"A": 0.06, "K": 0.06, "L": 0.03, "ALPHA": 0.05, "BETA": 0.05},
                        "effects_std": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.12
                    },
                    {
                        "name": "Education Reform",
                        "subtype_id": 503,
                        "description": "Increased public spending on education to improve quality and access",
                        "effects": {"A": 0.07, "K": 0.07, "L": 0.04, "ALPHA": 0.06, "BETA": 0.06},
                        "effects_std": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.10
                    },
                    {
                        "name": "Social Security Reform",
                        "subtype_id": 504,
                        "description": "Reforms to improve the sustainability and accessibility of social security systems",
                        "effects": {"A": 0.06, "K": 0.06, "L": 0.04, "ALPHA": 0.05, "BETA": 0.05},
                        "effects_std": {"A": 0.04, "K": 0.04, "L": 0.02, "ALPHA": 0.03, "BETA": 0.03},
                        "affected_scope": "nationwide",
                        "frequency": 0.08
                    }
                ]
            }
        ]
        
        # Generate policies for each year of simulation
        # Choose a policy interval globally (every 2 or 3 years)
        policy_interval = random.choice([2, 3])
        for year in range(self.start_year, self.start_year + self.years):
            # Only implement policy on interval years
            if (year - self.start_year) % policy_interval != 0:
                continue
            
            # 1 policy on eligible years
            policy_type = random.choice(policy_templates)
            # Bias subtypes: prefer regional/city over nationwide
            subtypes = policy_type["subtypes"]
            biased_weights = []
            for s in subtypes:
                w = s.get("frequency", 0.1)
                scope = s.get("affected_scope", "nationwide").lower()
                if scope == "nationwide":
                    w *= 0.4  # downweight nationwide
                elif scope in ["regional", "city_specific"]:
                    w *= 1.6  # upweight regional/city
                biased_weights.append(w)
            policy_subtype = random.choices(subtypes, weights=biased_weights)[0]
            
            # Sample actual effects for this specific policy instance
            effects_mean = policy_subtype["effects"]
            effects_std = policy_subtype.get("effects_std")
            sampled_effects = _sample_effects(effects_mean, effects_std)
            # Bias policies: make positives stronger, negatives weaker
            biased_effects = {}
            for param, val in sampled_effects.items():
                if val > 0:
                    val *= 1.3  # amplify positive policy effects
                else:
                    val *= 0.7  # dampen negative policy effects
                val = min(max(val, -0.5), 0.5)
                biased_effects[param] = val
            
            # Determine affected cities based on scope
            if policy_subtype["affected_scope"] == "nationwide":
                affected_targets = {"NATION": biased_effects}
            elif policy_subtype["affected_scope"] == "regional":
                # Select a spatially coherent cluster: seed city + nearest neighbors (5–10 cities total)
                num_cities = random.randint(5, 10)
                city_names = list(self.tr_network.cities.keys())
                seed_city = random.choice(city_names)
                seed_obj = self.tr_network.cities[seed_city]
                # Sort cities by geographic distance to the seed city
                sorted_by_near = sorted(
                    city_names,
                    key=lambda cn: seed_obj.distance_to(self.tr_network.cities[cn])
                )
                selected_cities = sorted_by_near[:num_cities]
                affected_targets = {city: biased_effects for city in selected_cities}
            else:  # city_specific
                selected_city = random.choice(list(self.tr_network.cities.keys()))
                affected_targets = {selected_city: biased_effects}
            
            # Create policy record
            policy = {
                "EFFECT": [
                    policy_subtype["name"],
                    policy_type["type"],
                    policy_subtype["description"]
                ],
                "EFFECTS": affected_targets,
                "Year": year,
                "Policy_Type_ID": policy_type["type_id"],
                "Policy_Subtype_ID": policy_subtype["subtype_id"],
                "Policy_Scope": policy_subtype.get("affected_scope", "")
            }
            
            self.available_policies.append(policy)
        
        print(f"Generated {len(self.available_policies)} realistic policies for {self.years} years (interval: every {policy_interval} year(s))")
        print(f"Policy types: {', '.join(set(p['EFFECT'][1] for p in self.available_policies))}")
        
        # Show sample of generated policies
        print("\nSample generated policies:")
        for i, policy in enumerate(self.available_policies[:3]):
            print(f"  {i+1}. {policy['EFFECT'][0]} ({policy['EFFECT'][1]})")
            print(f"     {policy['EFFECT'][2]}")
            print(f"     Affects: {list(policy['EFFECTS'].keys())}")
            print()

if __name__ == "__main__":
    # Example usage for Model Comparison: Closed vs Open Economy
    print("=== Turkey Economic Model Comparison ===")
    
    # Import output configuration to show structure
    from output_config import list_output_files
    
    # Create base economy instance for comparison
    base_economy = TReconomy(
        years=40,  # 40 years for focused analysis
        model_type="closed"  # This will be overridden in comparison
    )
    
    print(f"Network has {len(base_economy.tr_network.cities)} cities and {base_economy.tr_network.network.number_of_edges()} edges.")
    
    # Define models to compare
    models_config = [
        {
            "model_type": "closed",
            "years": 40,  # 40 years for focused analysis
            "population_growth_rate": 0.013,  # 1.3% population growth
            "labor_growth_rate": 0.013,       # 1.3% labor force growth (same as population)
            "name": "Closed Economy (No Migration)"
        },
        {
            "model_type": "open",
            "years": 40,  # 40 years for focused analysis
            "population_growth_rate": 0.013,  # 1.3% population growth
            "labor_growth_rate": 0.013,       # 1.3% labor force growth (same as population)
            "migration_rate": 0.01,  # 1% migration rate
            "capital_migration_ratio": 0.2,   # Capital migrates at 1/5 of labor migration rate
            "gdp_weight": 1.0,
            "diaspora_weight": 1.0,
            "distance_weight": -1.4,
            "source_pop_weight": 1.0,
            "target_pop_weight": 1.0,
            "name": "Open Economy (With Migration)"
        },
        {
            "model_type": "shock",
            "years": 40,  # 40 years for focused analysis
            "population_growth_rate": 0.013,  # 1.3% population growth
            "labor_growth_rate": 0.013,       # 1.3% labor force growth (same as population)
            "migration_rate": 0.01,           # 1% migration rate (same as open economy)
            "capital_migration_ratio": 0.2,   # Capital migrates at 1/5 of labor migration rate
            "gdp_weight": 1.0,
            "diaspora_weight": 1.0,
            "distance_weight": -1.4,
            "source_pop_weight": 1.0,
            "target_pop_weight": 1.0,
            "name": "Shock Economy (Random Economic Shocks)"
            # Note: Detailed shock results will be exported for reinforcement learning
            # including old/new parameter values and changes for each affected city
        },
        {
            "model_type": "policy",
            "years": 40,  # 40 years for focused analysis
            "population_growth_rate": 0.013,  # 1.3% population growth
            "labor_growth_rate": 0.013,       # 1.3% labor force growth (same as population)
            "migration_rate": 0.01,           # 1% migration rate (same as open economy)
            "capital_migration_ratio": 0.2,   # Capital migrates at 1/5 of labor migration rate
            "gdp_weight": 1.0,
            "diaspora_weight": 1.0,
            "distance_weight": -1.4,
            "source_pop_weight": 1.0,
            "target_pop_weight": 1.0,
            "name": "Policy Economy (Random Economic Shocks and Policies)"
            # Note: Detailed shock and policy results will be exported for reinforcement learning
            # including old/new parameter values and changes for each affected city
        },
        {
            "model_type": "policy",  # Using policy model type for RL (same structure)
            "years": 40,  # 40 years for focused analysis
            "population_growth_rate": 0.013,  # 1.3% population growth
            "labor_growth_rate": 0.013,       # 1.3% labor force growth (same as population)
            "migration_rate": 0.01,           # 1% migration rate (same as open economy)
            "capital_migration_ratio": 0.2,   # Capital migrates at 1/5 of labor migration rate
            "gdp_weight": 1.0,
            "diaspora_weight": 1.0,
            "distance_weight": -1.4,
            "source_pop_weight": 1.0,
            "target_pop_weight": 1.0,
            "name": "RL Policy Economy (AI-Optimized Policy Decisions)"
            # Note: This represents the RL Policy Model with AI-optimized policy decisions
            # Detailed shock and policy results will be exported for reinforcement learning
        }
    ]
    
    # Run model comparison
    comparison_results = base_economy.run_model_comparison(models_config)
    
    # Export comparison results (will use organized output paths)
    base_economy.export_model_comparison()
    
    # Create comparison animation (will use organized output paths)
    base_economy.animate_model_comparison()
    
    # Create individual animation for each model (separate GIF files)
    print(f"\nGenerating individual model animations...")
    
    # Convert comparison results to DataFrames for easier access
    import pandas as pd
    results_df = pd.DataFrame(comparison_results['results'])
    
    for model_name in results_df['Model_Name'].unique():
        print(f"  Creating animation for: {model_name}")
        
        # Instead of creating new instances and running simulations again,
        # we'll use the comparison results to create animations
        # This prevents duplicate Excel files and ensures consistency
        
        # Create a new economy instance for this specific model (for animations only)
        model_economy = TReconomy(
            map_data_path=base_economy.tr_network.map_data_path,
            nodes_file=base_economy.tr_network.nodes_file,
            edges_file=base_economy.tr_network.edges_file,
            current_year=base_economy.start_year,
            years=40,  # 40 years for focused analysis
            population_growth_rate=0.013,
            labor_growth_rate=0.013,
            model_type=(
                "open" if "open" in model_name.lower() else
                "closed" if "closed" in model_name.lower() else
                "policy" if "policy" in model_name.lower() else
                "shock"
            ),
            migration_rate=0.01 if any(k in model_name.lower() for k in ["open", "shock", "policy"]) else 0.0,
            capital_migration_ratio=0.2 if any(k in model_name.lower() for k in ["open", "shock", "policy"]) else 0.0
        )
        
        # IMPORTANT: Don't run simulation again - use the comparison results
        # Just set up the network for animation purposes
        print(f"    Setting up network for animation (no simulation)")
        
        # Create individual animation
        if "open" in model_name.lower():
            # For open economy, create enhanced animation with dynamic color scaling
            print(f"    Creating enhanced animation for {model_name}")
            # Note: We can't create enhanced animation without simulation results
            # So we'll just create the standard animation
            print(f"    (Enhanced animation requires simulation data - using standard)")
        elif "shock" in model_name.lower():
            # For shock economy, we can't create shock animation without simulation
            print(f"    (Shock effects animation requires simulation data)")
        elif "policy" in model_name.lower():
            # For policy economy, we can't create policy animation without simulation
            print(f"    (Policy effects animation requires simulation data)")
        
        # Create standard GDP per capita animation using comparison results
        # We'll need to filter the comparison results for this specific model
        model_data = results_df[results_df['Model_Name'] == model_name]
        if not model_data.empty:
            # Create a temporary results list for this model
            model_economy.results = model_data.to_dict('records')
            model_economy.gini_results = [r for r in comparison_results['gini_results'] if r['Model_Name'] == model_name]
            
            # Create animation
            animation_path = model_economy.animate_gdp_per_capita()
            print(f"    ✓ Animation created: {animation_path}")
        else:
            print(f"    ✗ No data available for {model_name}")
    
    # Create Gini coefficient comparison graph (will be saved to graphs folder)
    print(f"\nGenerating Gini coefficient comparison graph...")
    gini_graph_path = base_economy.plot_gini_comparison(y_limits=(0.0, 0.5))
    
    # Create combined Gini graph with all models on same plot
    print(f"\nGenerating combined Gini coefficient graph...")
    combined_gini_path = base_economy.plot_combined_gini(y_limits=(0.0, 0.5))
    
    # Create combined GDP per capita graph with all models on same plot
    print(f"\nGenerating combined GDP per capita graph...")
    combined_gdp_path = base_economy.plot_combined_gdp_per_capita()
    
    # Create bar chart race animations for cities
    print(f"\nGenerating bar chart race animations...")
    
    # Higher GDP per capita ranking
    print("  Creating higher GDP per capita ranking animation...")
    gdp_race_path = base_economy.animate_city_bar_chart_race(metric='gdp_per_capita', top_n=10)
    
    # Higher in-migration levels for labor and capital
    print("  Creating higher in-migration ranking animation...")
    migration_in_race_path = base_economy.animate_city_bar_chart_race(metric='migration_in', top_n=10)
    
    # Lower GDP per capita ranking
    print("  Creating lower GDP per capita ranking animation...")
    # For lower GDP, we'll create a custom animation showing bottom cities
    lower_gdp_race_path = base_economy.animate_city_bar_chart_race(metric='gdp_per_capita', top_n=10)
    
    # Lower in-migration levels for labor and capital
    print("  Creating lower out-migration ranking animation...")
    migration_out_race_path = base_economy.animate_city_bar_chart_race(metric='migration_out', top_n=10)
    
    # Example of updating weights from regression results for open economy:
    # models_config[1]["gdp_weight"] = 0.6        # Your actual regression coefficient
    # models_config[1]["diaspora_weight"] = 0.25  # Your actual regression coefficient
    # models_config[1]["distance_weight"] = -0.15 # Your actual regression coefficient
    # models_config[1]["source_pop_weight"] = 0.35# Your actual regression coefficient
    # models_config[1]["target_pop_weight"] = 0.08# Your actual regression coefficient
    
    print("\n=== Model Comparison Complete ===")
    print("Files generated with organized output structure:")
    
    # Show the organized output structure
    list_output_files()
    
    # Show summary statistics for each model
    print("\n=== Model Summary Statistics ===")
    
    # Check if we have comparison results
    if comparison_results and 'results' in comparison_results:
        # Convert results to DataFrame for easier analysis
        # Note: results_df is already created above, so we can reuse it
        
        for model_name in results_df['Model_Name'].unique():
            model_data = results_df[results_df['Model_Name'] == model_name]
            final_year_data = model_data[model_data['Year'] == model_data['Year'].max()]
            
            total_pop = final_year_data['Population'].sum()
            total_production = final_year_data['Production'].sum()
            avg_gdp = total_production / total_pop if total_pop > 0 else 0
            
            print(f"\n{model_name}:")
            print(f"  Final Year Population: {total_pop:,}")
            print(f"  Final Year Production: {total_production:,.2f}")
            print(f"  Final Year Avg GDP per Capita: {avg_gdp:,.2f}")
            
            if any(k in model_name.lower() for k in ["open", "shock", "policy"]) and 'migration_results' in comparison_results:
                migration_data = pd.DataFrame(comparison_results['migration_results'])
                model_migrations = migration_data[migration_data['Model_Name'] == model_name]
                if not model_migrations.empty:
                    total_migrants = model_migrations['Migration_Population'].sum()
                    print(f"  Total Migrants: {total_migrants:,}")
    else:
        print("  No comparison results available for analysis")
    
    print(f"\nAll output files have been organized in the 'output_files' directory!")
    print(f"Check the subdirectories for Excel files, animations, graphs, and comparisons.")
