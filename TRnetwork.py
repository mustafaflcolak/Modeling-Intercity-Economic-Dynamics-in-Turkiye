import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import the City class
from city import City


class TRnetwork:
    """
    A class to visualize Turkey's cities on a map with network-based coloring.
    Uses City class from city.py to define nodes with economic and demographic parameters.
    """
    
    def __init__(self, map_data_path: str = "harita_dosyaları", nodes_file: str = "datalar/network_nodes.xlsx", edges_file: str = "datalar/network_edge_weights.xlsx", current_year: int = 2024):
        """
        Initialize the Turkey Map Visualizer.
        
        Args:
            map_data_path (str): Path to the map data files
            nodes_file (str): Path to the nodes data file
            edges_file (str): Path to the edges data file
            current_year (int): Initial year for the simulation (default: 2023)
        """
        self.map_data_path = map_data_path
        self.nodes_file = nodes_file
        self.edges_file = edges_file
        self.current_year = current_year
        self.world = None
        self.turkey = None
        self.provinces = None
        self.node_data = None
        self.edge_data = None
        self.network = None
        self.cities = {}  # Dictionary to store City objects
        
    def load_map_data(self):
        """Load the map data for Turkey."""
        try:
            # Load world map data
            world_path = os.path.join(self.map_data_path, "ne_10m_admin_0_countries.shp")
            if os.path.exists(world_path):
                self.world = gpd.read_file(world_path)
                self.turkey = self.world[self.world["ADMIN"] == "Turkey"]
            else:
                print(f"Warning: World map file not found at {world_path}")
                self.turkey = None
            
            # Load provinces data
            provinces_path = os.path.join(self.map_data_path, "gadm36_TUR_1.shp")
            if os.path.exists(provinces_path):
                self.provinces = gpd.read_file(provinces_path)
                # Set province names
                self.provinces["NAME_1"] = [
                    'ADANA','ADIYAMAN','AFYONKARAHISAR','AGRI','AKSARAY','AMASYA','ANKARA','ANTALYA',
                    'ARDAHAN','ARTVIN','AYDIN','BALIKESIR','BARTIN','BATMAN','BAYBURT','BILECIK',
                    'BINGOL','BITLIS','BOLU','BURDUR','BURSA','CANAKKALE','CANKIRI','CORUM',
                    'DENIZLI','DIYARBAKIR','DUZCE','EDIRNE','ELAZIG','ERZINCAN','ERZURUM',
                    'ESKISEHIR','GAZIANTEP','GIRESUN','GUMUSHANE','HAKKARI','HATAY','IGDIR',
                    'ISPARTA','ISTANBUL','IZMIR','KAHRAMANMARAS','KARABUK','KARAMAN','KARS',
                    'KASTAMONU','KAYSERI','KILIS','KIRIKKALE','KIRKLARELI','KIRSEHIR','KOCAELI',
                    'KONYA','KUTAHYA','MALATYA','MANISA','MARDIN','MERSIN','MUGLA','MUS',
                    'NEVSEHIR','NIGDE','ORDU','OSMANIYE','RIZE','SAKARYA','SAMSUN','SANLIURFA',
                    'SIIRT','SINOP','SIRNAK','SIVAS','TEKIRDAG','TOKAT','TRABZON','TUNCELI',
                    'USAK','VAN','YALOVA','YOZGAT','ZONGULDAK'
                ]
            else:
                print(f"Warning: Provinces map file not found at {provinces_path}")
                self.provinces = None
                
        except Exception as e:
            print(f"Error loading map data: {e}")
    
    def load_network_data(self, nodes_file: Optional[str] = None, edges_file: Optional[str] = None):
        """
        Load network data from Excel files.
        
        Args:
            nodes_file (str): Path to nodes data file
            edges_file (str): Path to edges data file
        """
        try:
            # Use instance variables if not provided
            if nodes_file is None:
                nodes_file = self.nodes_file
            if edges_file is None:
                edges_file = self.edges_file

            # Load node data
            if os.path.exists(nodes_file):
                self.node_data = pd.read_excel(nodes_file)
                print(f"Loaded node data with {len(self.node_data)} cities")
            else:
                print(f"Warning: Node data file not found at {nodes_file}")
                self.node_data = None

            # Load edge data
            if os.path.exists(edges_file):
                self.edge_data = pd.read_excel(edges_file)
                print(f"Loaded edge data with {len(self.edge_data)} rows")
            else:
                print(f"Warning: Edge data file not found at {edges_file}")
                self.edge_data = None
                
        except Exception as e:
            print(f"Error loading network data: {e}")
    
    def update_time(self, new_year: int) -> None:
        """
        Update the current year and update all cities to the new year.
        
        Args:
            new_year (int): New year for the simulation
        """
        self.current_year = new_year
        # Update year for all cities
        for city in self.cities.values():
            city.update_year(new_year)
        print(f"Time updated to year {new_year}")
    
    def get_current_year(self) -> int:
        """
        Get the current year of the simulation.
        
        Returns:
            Current year
        """
        return self.current_year
    
    def advance_time(self, years: int = 1) -> None:
        """
        Advance time by a specified number of years.
        
        Args:
            years (int): Number of years to advance (default: 1)
        """
        new_year = self.current_year + years
        self.update_time(new_year)
    
    def create_city_objects(self):
        """Create City objects from the loaded node data."""
        if self.node_data is None:
            print("Cannot create city objects: no node data loaded")
            return
        
        try:
            for _, row in self.node_data.iterrows():
                city_name = row["CITY"]
                
                # Create City object with basic parameters
                city = City(
                    name=city_name,
                    longitude=row['LOCATION_LON'],
                    latitude=row['LOCATION_LAT'],
                    population=row.get('POPULATION', 0),  # Population from data file
                    capital_stock=row.get('CAPITAL_STOCK', 100000),  # Default capital if not available
                    alpha=row.get('ALPHA', 0.3),  # Default Cobb-Douglas alpha
                    beta=row.get('BETA', 0.7),    # Default Cobb-Douglas beta
                    A=row.get('A', 1.0),          # Default total factor productivity
                    year=self.current_year        # Set current year
                )
                
                # Store city object
                self.cities[city_name] = city
                
            print(f"Created {len(self.cities)} City objects for year {self.current_year}")
            
        except Exception as e:
            print(f"Error creating city objects: {e}")
    
    def create_network(self):
        """Create NetworkX graph from the loaded data using City objects."""
        if self.node_data is None or self.edge_data is None:
            print("Cannot create network: missing node or edge data")
            return
        
        # Create city objects if not already created
        if not self.cities:
            self.create_city_objects()
        
        try:
            # Create NetworkX graph
            self.network = nx.DiGraph()
            
            # Add nodes with City objects and their attributes
            for city_name, city_obj in self.cities.items():
                # Get all city information as dictionary
                city_info = city_obj.get_all_info()
                
                # Add node with position and all city attributes
                self.network.add_node(
                    city_name, 
                    pos=(city_obj.longitude, city_obj.latitude),
                    city_object=city_obj,
                    **city_info
                )
            
            # Process edge data
            edge_df = self.edge_data.melt(
                id_vars='CITY', 
                var_name='CITY_born', 
                value_name='LEVEL'
            )
            edge_df = edge_df.rename(columns={
                "CITY": "CITY_TARGET",
                "CITY_born": "CITY_SOURCE"
            })
            
            # Add weight column for percentage shares
            edge_df['WEIGHT'] = edge_df['LEVEL']
            
            # Calculate percentage shares for each source city
            for source_city in edge_df['CITY_SOURCE'].unique():
                source_mask = edge_df['CITY_SOURCE'] == source_city
                source_total = edge_df.loc[source_mask, 'LEVEL'].sum()
                
                if source_total > 0:
                    # Calculate percentage share for each target city from this source
                    edge_df.loc[source_mask, 'WEIGHT'] = (edge_df.loc[source_mask, 'LEVEL'] / source_total) * 100
            
            # Add edges and update city demographics
            for _, row in edge_df.iterrows():
                source_city = row["CITY_SOURCE"]
                target_city = row["CITY_TARGET"]
                weight = row["WEIGHT"]
                
                if weight > 0:  # Only add edges with positive weights
                    self.network.add_edge(source_city, target_city, weight=weight)
                    
                    # Update target city's demographics
                    if target_city in self.cities and source_city in self.cities:
                        # Add labor demographic from edges data
                        self.cities[target_city].add_labor_demographic(source_city, int(row["LEVEL"]))

            # Note: Population comes from node data, labor force comes from edges data
            # No need to reconcile them as they are separate concepts
            print(f"Created network with {self.network.number_of_nodes()} nodes and {self.network.number_of_edges()} edges")
            print(f"Population data: from node data (total residents)")
            print(f"Labor force data: from edges data (working people)")
            
        except Exception as e:
            print(f"Error creating network: {e}")
    
    def calculate_node_metrics(self, metric: str = 'out_degree') -> Dict[str, float]:
        """
        Calculate metrics for each node.
        
        Args:
            metric (str): Metric to calculate ('out_degree', 'in_degree', 'total_weight', 
                         'population', 'gdp_per_capita', 'immigration_rate', 'production')
            
        Returns:
            Dict[str, float]: Dictionary mapping city names to metric values
        """
        if self.network is None:
            return {}
        
        metrics = {}
        
        if metric == 'out_degree':
            metrics = dict(self.network.out_degree())
        elif metric == 'in_degree':
            metrics = dict(self.network.in_degree())
        elif metric == 'total_weight':
            for city in self.network.nodes():
                total_weight = sum(
                    self.network[city][neighbor]['weight'] 
                    for neighbor in self.network.successors(city)
                )
                metrics[city] = total_weight
        elif metric == 'labor':
            for city_name in self.network.nodes():
                city_obj = self.cities.get(city_name)
                if city_obj:
                    metrics[city_name] = city_obj.labor_force
                else:
                    metrics[city_name] = 0.0
        elif metric == 'gdp_per_capita':
            for city_name in self.network.nodes():
                city_obj = self.cities.get(city_name)
                if city_obj and city_obj.population > 0:
                    try:
                        # Calculate GDP per capita using population
                        production = city_obj.calculate_production()
                        metrics[city_name] = production / city_obj.population
                    except (ZeroDivisionError, ValueError):
                        metrics[city_name] = 0.0
                else:
                    metrics[city_name] = 0.0
        elif metric == 'immigration_rate':
            for city_name in self.network.nodes():
                city_obj = self.cities.get(city_name)
                if city_obj and city_obj.labor_force > 0:
                    metrics[city_name] = city_obj.get_total_labor_immigrants() / city_obj.labor_force
                else:
                    metrics[city_name] = 0.0
        elif metric == 'production':
            for city_name in self.network.nodes():
                city_obj = self.cities.get(city_name)
                if city_obj:
                    try:
                        metrics[city_name] = city_obj.calculate_production()
                    except (ZeroDivisionError, ValueError):
                        metrics[city_name] = 0.0
                else:
                    metrics[city_name] = 0.0
        elif metric == 'capital_stock':
            for city_name in self.network.nodes():
                city_obj = self.cities.get(city_name)
                if city_obj:
                    metrics[city_name] = city_obj.capital_stock
                else:
                    metrics[city_name] = 0.0
        else:
            print(f"Unknown metric: {metric}")
            return {}
        
        return metrics
    
    def visualize_turkey_map(self, 
                            metric: str = 'gdp_per_capita',
                            figsize: tuple = (20, 16),
                            cmap: str = 'RdYlGn',
                            title: str = None,
                            save_path: Optional[str] = None):
        """
        Visualize Turkey map with provinces colored based on network metrics.
        
        Args:
            metric (str): Metric to use for coloring ('out_degree', 'in_degree', 'total_weight',
                         'population', 'gdp_per_capita', 'immigration_rate', 'production')
            figsize (tuple): Figure size
            cmap (str): Colormap for visualization
            title (str): Plot title (if None, will use default with year)
            save_path (Optional[str]): Path to save the plot
        """
        if self.provinces is None:
            print("Cannot visualize: provinces data not loaded")
            return
        
        if self.network is None:
            print("Cannot visualize: network not created")
            return
        
        # Set default title with year if not provided
        if title is None:
            title = f"Turkey Cities Network Visualization - {self.current_year}"
        
        # Calculate metrics
        metrics = self.calculate_node_metrics(metric)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create a mapping from province names to metric values
        province_values = {}
        for city_name, value in metrics.items():
            # Find the province that contains this city
            for idx, province_name in enumerate(self.provinces['NAME_1']):
                if city_name.upper() in province_name.upper() or province_name.upper() in city_name.upper():
                    province_values[province_name] = value
                    break
        
        # Create a list of values for each province in the same order as the provinces dataframe
        province_value_list = []
        for province_name in self.provinces['NAME_1']:
            if province_name in province_values:
                province_value_list.append(province_values[province_name])
            else:
                province_value_list.append(0.0)  # Default value for provinces without data
        
        # Add the values as a new column to the provinces dataframe
        self.provinces['metric_values'] = province_value_list
        
        # Normalize values for coloring
        province_values_array = np.array(province_value_list)
        if province_values_array.max() > province_values_array.min():
            normalized_values = (province_values_array - province_values_array.min()) / (province_values_array.max() - province_values_array.min())
        else:
            normalized_values = np.ones_like(province_values_array) * 0.5
        
        # Plot Turkey provinces with gradient colors
        self.provinces.plot(
            ax=ax, 
            column='metric_values',
            cmap=cmap,
            edgecolor='white', 
            linewidth=0.5,
            legend=True,
            legend_kwds={'label': f'{metric.replace("_", " ").title()}', 'orientation': 'vertical', 'shrink': 0.6, 'aspect': 20}
        )
        
        # Add city labels without boxes
        for city in self.network.nodes():
            if city in metrics:
                pos = self.network.nodes[city]['pos']
                ax.annotate(
                    city,
                    xy=(pos[0], pos[1]),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=10,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    color='black'
                )
        
        # Customize plot
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        # Remove axis labels and grids
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set axis limits to focus on Turkey
        ax.set_xlim(25, 45)
        ax.set_ylim(35, 43)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map saved to {save_path}")
        
        plt.show()
    
    def get_city_summary(self, city_name: str) -> Dict:
        """
        Get a comprehensive summary of a specific city.
        
        Args:
            city_name (str): Name of the city
            
        Returns:
            Dict: Dictionary containing all city information
        """
        if city_name in self.cities:
            city_obj = self.cities[city_name]
            return city_obj.get_all_info()
        return {}
    
    def calculate_gini_coefficient(self) -> float:
        """
        Calculate Gini coefficient for income inequality using GDP per capita data.
        
        Returns:
            float: Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if not self.cities:
            return 0.0
        
        # Get GDP per capita for all cities with population
        gdp_per_capita_values = []
        for city_name, city_obj in self.cities.items():
            if city_obj.population > 0:
                try:
                    # Calculate GDP per capita using population
                    production = city_obj.calculate_production()
                    gdp_per_capita = production / city_obj.population
                    if gdp_per_capita > 0:
                        gdp_per_capita_values.append(gdp_per_capita)
                except:
                    continue
        
        if len(gdp_per_capita_values) < 2:
            return 0.0
        
        # Sort GDP per capita values in ascending order
        gdp_per_capita_values.sort()
        n = len(gdp_per_capita_values)
        
        # Calculate Gini coefficient using the formula:
        # G = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
        # where x_i are the sorted values and i is the rank (1 to n)
        
        sum_ranked_values = sum((i + 1) * gdp_per_capita_values[i] for i in range(n))
        sum_values = sum(gdp_per_capita_values)
        
        gini = (2 * sum_ranked_values - (n + 1) * sum_values) / (n * sum_values)
        
        return abs(gini)  # Ensure positive value
    
    def get_income_inequality_summary(self) -> Dict[str, float]:
        """
        Get a comprehensive summary of income inequality metrics.
        
        Returns:
            Dict: Dictionary containing income inequality statistics
        """
        if not self.cities:
            return {}
        
        # Get GDP per capita for all cities with population
        gdp_values = []
        city_names = []
        for city_name, city_obj in self.cities.items():
            if city_obj.population > 0:
                try:
                    # Calculate GDP per capita using population
                    production = city_obj.calculate_production()
                    gdp_per_capita = production / city_obj.population
                    if gdp_per_capita > 0:
                        gdp_values.append(gdp_per_capita)
                        city_names.append(city_name)
                except:
                    continue
        
        if len(gdp_values) < 2:
            return {
                'gini_coefficient': 0.0,
                'mean_gdp_per_capita': 0.0,
                'median_gdp_per_capita': 0.0,
                'min_gdp_per_capita': 0.0,
                'max_gdp_per_capita': 0.0,
                'std_gdp_per_capita': 0.0,
                'richest_city': 'N/A',
                'poorest_city': 'N/A'
            }
        
        # Calculate inequality metrics
        gini = self.calculate_gini_coefficient()
        mean_gdp = np.mean(gdp_values)
        median_gdp = np.median(gdp_values)
        min_gdp = np.min(gdp_values)
        max_gdp = np.max(gdp_values)
        std_gdp = np.std(gdp_values)
        
        # Find richest and poorest cities
        min_idx = np.argmin(gdp_values)
        max_idx = np.argmax(gdp_values)
        poorest_city = city_names[min_idx]
        richest_city = city_names[max_idx]
        
        return {
            'gini_coefficient': gini,
            'mean_gdp_per_capita': mean_gdp,
            'median_gdp_per_capita': median_gdp,
            'min_gdp_per_capita': min_gdp,
            'max_gdp_per_capita': max_gdp,
            'std_gdp_per_capita': std_gdp,
            'richest_city': richest_city,
            'poorest_city': poorest_city
        }
    
    def export_city_data_to_excel(self, filename: str = None):
        """
        Export all city information to an Excel file with multiple sheets.
        
        Args:
            filename (str): Name of the Excel file to create (if None, will use default with year)
        """
        if not self.cities:
            print("No city data available to export")
            return
        
        # Set default filename with year if not provided
        if filename is None:
            filename = f"turkey_cities_data_{self.current_year}.xlsx"
        
        try:
            # Create a writer object
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                
                # Sheet 1: Basic city information
                # Note: Population and Labor Force are separate concepts:
                # - Population: Total residents (from node data)
                # - Labor Force: Working people (from edges data)
                basic_info = []
                for city_name, city_obj in self.cities.items():
                    basic_data = {
                        'City_Name': city_name,
                        'Year': city_obj.year,
                        'Longitude': city_obj.longitude,
                        'Latitude': city_obj.latitude,
                        'Population': city_obj.population,
                        'Labor_Force': city_obj.labor_force,
                        'Capital_Stock': city_obj.capital_stock,
                        'Alpha': city_obj.alpha,
                        'Beta': city_obj.beta,
                        'Total_Factor_Productivity': city_obj.A,
                        'Total_Labor_Immigrants': city_obj.get_total_labor_immigrants(),
                        'Labor_Native': city_obj.get_labor_native(),
                        'Labor_Immigration_Rate': city_obj.get_total_labor_immigrants() / city_obj.labor_force if city_obj.labor_force > 0 else 0
                    }
                    
                    # Add calculated economic values
                    try:
                        basic_data['Production_Output'] = city_obj.calculate_production()
                        basic_data['GDP_per_Capita'] = city_obj.calculate_gdp_per_capita()
                    except:
                        basic_data['Production_Output'] = 0.0
                        basic_data['GDP_per_Capita'] = 0.0
                    
                    basic_info.append(basic_data)
                
                basic_df = pd.DataFrame(basic_info)
                basic_df.to_excel(writer, sheet_name='Basic_City_Info', index=False)
                
                # Sheet 2: Labor Demographic breakdown
                demographic_data = []
                for city_name, city_obj in self.cities.items():
                    for demo in city_obj.demographics:
                        demographic_data.append({
                            'City_Name': city_name,
                            'Year': city_obj.year,
                            'Origin_City': demo.origin_city,
                            'Population_Count': demo.population_count
                        })
                
                if demographic_data:
                    demo_df = pd.DataFrame(demographic_data)
                    demo_df.to_excel(writer, sheet_name='Labor_Demographics', index=False)
                else:
                    # Create empty sheet if no demographic data
                    pd.DataFrame(columns=['City_Name', 'Year', 'Origin_City', 'Population_Count']).to_excel(
                        writer, sheet_name='Labor_Demographics', index=False
                    )
                
                # Sheet 3: Network metrics
                network_metrics = []
                metrics_to_calculate = ['out_degree', 'in_degree', 'total_weight', 'labor', 
                                      'gdp_per_capita', 'immigration_rate', 'production', 'capital_stock']
                
                for metric in metrics_to_calculate:
                    metric_values = self.calculate_node_metrics(metric)
                    for city_name, value in metric_values.items():
                        network_metrics.append({
                            'City_Name': city_name,
                            'Year': self.current_year,
                            'Metric': metric.replace('_', ' ').title(),
                            'Value': value
                        })
                
                network_df = pd.DataFrame(network_metrics)
                network_df.to_excel(writer, sheet_name='Network_Metrics', index=False)
                
                # Sheet 4: Network summary
                summary = self.get_network_summary()
                summary_data = []
                for key, value in summary.items():
                    summary_data.append({
                        'Year': self.current_year,
                        'Metric': key.replace('_', ' ').title(),
                        'Value': value
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Network_Summary', index=False)
                
                # Sheet 5: Income Inequality Analysis
                inequality_summary = self.get_income_inequality_summary()
                inequality_data = []
                for key, value in inequality_summary.items():
                    inequality_data.append({
                        'Year': self.current_year,
                        'Metric': key.replace('_', ' ').title(),
                        'Value': value
                    })
                
                inequality_df = pd.DataFrame(inequality_data)
                inequality_df.to_excel(writer, sheet_name='Income_Inequality', index=False)
                
                # Sheet 6: Edge data (if available)
                if self.edge_data is not None:
                    edge_data_with_year = self.edge_data.copy()
                    edge_data_with_year['Year'] = self.current_year
                    edge_data_with_year.to_excel(writer, sheet_name='Edge_Data', index=False)
                
                # Sheet 7: Node data (if available)
                if self.node_data is not None:
                    node_data_with_year = self.node_data.copy()
                    node_data_with_year['Year'] = self.current_year
                    node_data_with_year.to_excel(writer, sheet_name='Node_Data', index=False)
            
            print(f"City data exported successfully to {filename}")
            print(f"File contains {len(self.cities)} cities with comprehensive information for year {self.current_year}")
            return True
            
        except Exception as e:
            print(f"Error exporting data to Excel: {e}")
            return False
    
    def get_network_summary(self) -> Dict:
        """
        Get a summary of the entire network with city statistics.
        
        Returns:
            Dict: Dictionary containing network and city statistics
        """
        if not self.cities:
            return {}
        
        # Filter cities with positive labor force for calculations
        # Note: Population and labor force are separate concepts:
        # - Population: Total residents (from node data)
        # - Labor force: Working people (from edges data)
        cities_with_population = [city for city in self.cities.values() if city.labor_force > 0]
        
        # Calculate total production with error handling
        total_production = 0.0
        for city in self.cities.values():
            try:
                total_production += city.calculate_production()
            except (ZeroDivisionError, ValueError):
                # Skip cities that can't calculate production
                continue
        
        summary = {
            'total_cities': len(self.cities),
            'total_population': sum(city.population for city in self.cities.values()),
            'total_labor_force': sum(city.labor_force for city in self.cities.values()),
            'total_capital': sum(city.capital_stock for city in self.cities.values()),
            'total_production': total_production,
            'total_labor_immigrants': sum(city.get_total_labor_immigrants() for city in self.cities.values()),
            'network_edges': self.network.number_of_edges() if self.network else 0
        }
        
        # Calculate average GDP per capita only for cities with population
        if cities_with_population:
            try:
                gdp_values = []
                for city in cities_with_population:
                    try:
                        gdp_values.append(city.calculate_gdp_per_capita())
                    except (ZeroDivisionError, ValueError):
                        continue
                summary['average_gdp_per_capita'] = np.mean(gdp_values) if gdp_values else 0.0
            except:
                summary['average_gdp_per_capita'] = 0.0
        else:
            summary['average_gdp_per_capita'] = 0.0
        
        return summary
    
    def run_demo(self):
        """Run a complete demonstration of the visualization."""
        print("Starting Turkey Map Visualization Demo with City Objects...")
        print(f"Initial year: {self.current_year}")
        
        # Load or create data
        self.load_map_data()
        self.load_network_data()
        
        # If no real data, create sample data
        if self.node_data is None or self.edge_data is None:
            print("Error: No real data files found. Cannot run demo.")
            return
        
        # Create network with City objects
        self.create_network()
        
        # Print network summary
        summary = self.get_network_summary()
        print(f"\nNetwork Summary for {self.current_year}:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Print income inequality summary
        inequality_summary = self.get_income_inequality_summary()
        print(f"\nIncome Inequality Analysis for {self.current_year}:")
        print(f"  Gini Coefficient: {inequality_summary.get('gini_coefficient', 0):.4f}")
        print(f"  Mean GDP per Capita: {inequality_summary.get('mean_gdp_per_capita', 0):,.2f}")
        print(f"  Median GDP per Capita: {inequality_summary.get('median_gdp_per_capita', 0):,.2f}")
        print(f"  Min GDP per Capita: {inequality_summary.get('min_gdp_per_capita', 0):,.2f}")
        print(f"  Max GDP per Capita: {inequality_summary.get('max_gdp_per_capita', 0):,.2f}")
        print(f"  Richest City: {inequality_summary.get('richest_city', 'N/A')}")
        print(f"  Poorest City: {inequality_summary.get('poorest_city', 'N/A')}")
        
        # Export city data to Excel
        print(f"\nExporting city data to Excel for year {self.current_year}...")
        self.export_city_data_to_excel()
        
        # Visualize with different metrics
        metrics_to_show = ['out_degree', 'labor', 'gdp_per_capita', 'immigration_rate', 'production']
        
        for metric in metrics_to_show:
            print(f"\nVisualizing with {metric} metric for year {self.current_year}...")
            self.visualize_turkey_map(
                metric=metric,
                title=f"Turkey Cities Network - {metric.replace('_', ' ').title()} - {self.current_year}",
                save_path=f"turkey_map_{metric}_{self.current_year}.png"
            )


def main():
    """Main function to run the visualization."""
    # Example: allow user to specify files here if desired
    visualizer = TRnetwork(
        map_data_path="harita_dosyaları",
        nodes_file="datalar/network_nodes.xlsx",
        edges_file="datalar/network_edge_weights.xlsx",
        current_year=2023
    )
    # Run demo
    visualizer.run_demo()

if __name__ == "__main__":
    main() 