import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class City:
    """
    A class representing a city with geographical, demographic, and economic parameters.
    
    Attributes:
        name (str): Name of the city
        longitude (float): Longitude coordinate of the city
        latitude (float): Latitude coordinate of the city
        demographics (List[City.Demographics]): List of people from other cities living in this city
        alpha (float): Capital share parameter for Cobb-Douglas production function
        beta (float): Labor share parameter for Cobb-Douglas production function
        A (float): Total factor productivity for Cobb-Douglas production function
        population (int): Total population of the city (from node data)
        capital_stock (float): Capital stock for production function
        labor_force (float): Labor force for production function (from edges data)
    """

    @dataclass
    class Demographics:
        """Class to store demographic information about people from other cities"""
        origin_city: str
        population_count: int
        
        def __repr__(self):
            return f"{self.origin_city}: {self.population_count} people"

    def __init__(self, 
                 name: str,
                 longitude: float,
                 latitude: float,
                 demographics: Optional[List['City.Demographics']] = None,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 A: float = 1.0,
                 population: int = 0,
                 capital_stock: float = 0.0,
                 year: int = 2023):
        """
        Initialize a City object.
        
        Args:
            name: Name of the city
            longitude: Longitude coordinate (decimal degrees)
            latitude: Latitude coordinate (decimal degrees)
            demographics: List of demographic information for people from other cities
            alpha: Capital share parameter for Cobb-Douglas production function
            beta: Labor share parameter for Cobb-Douglas production function
            A: Total factor productivity for Cobb-Douglas production function
            population: Total population of the city from data file
            capital_stock: Capital stock for production function
            year: Year for the city data (default: 2023)
        """
        self.name = name
        self.longitude = longitude
        self.latitude = latitude
        self.demographics = demographics if demographics is not None else []
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.population = population  # Population from data file
        self.labor_force = 0  # Labor force will be built from demographics
        self.capital_stock = capital_stock
        self.year = year
    

    
    def get_population(self) -> int:
        """
        Get the population of the city from data file.
        
        Returns:
            Population count
        """
        return self.population
    
    def set_labor_force(self, labor_force: int) -> None:
        """
        Set the labor force value for the city.
        Note: Labor force is separate from population and comes from edges data.
        
        Args:
            labor_force: New labor force value
        """
        self.labor_force = labor_force
    
    def add_labor_demographic(self, origin_city: str, labor_count: int) -> None:
        """
        Add labor demographic information for people from another city (or native).
        Args:
            origin_city: Name of the origin city
            labor_count: Number of people from that city working here
        """
        # Always add or update the demographic group, including native labor
        for demo in self.demographics:
            if demo.origin_city == origin_city:
                demo.population_count += labor_count
                self.labor_force += labor_count
                return
        # Add new demographic entry
        self.demographics.append(self.Demographics(origin_city, labor_count))
        self.labor_force += labor_count
    
    def get_total_labor_immigrants(self) -> int:
        """
        Get the total number of labor immigrants (people from other cities working here).
        
        Returns:
            Total number of labor immigrants
        """
        return sum(demo.population_count for demo in self.demographics)
    
    def get_labor_native(self) -> int:
        """
        Get the native labor force (people born in this city working here).
        
        Returns:
            Native labor force count
        """
        return self.labor_force - self.get_total_labor_immigrants()
   

    
    def calculate_production(self) -> float:
        """
        Calculate output using Cobb-Douglas production function.
        
        Returns:
            Output value (Y = A * K^alpha * L^beta)
        """
        if self.labor_force <= 0:
            return 0.0
        
        return self.A * (self.capital_stock ** self.alpha) * (self.labor_force ** self.beta)
    
    def calculate_gdp_per_capita(self) -> float:
        """
        Calculate GDP per capita by dividing total production by total population.
        
        Returns:
            GDP per capita (production output / total population)
        """
        if self.labor_force <= 0:
            raise ValueError("Cannot calculate GDP per capita with zero population")
        
        total_production = self.calculate_production()
        return total_production / self.population
    
    def distance_to(self, other_city: 'City') -> float:
        """
        Calculate distance to another city using Haversine formula.
        
        Args:
            other_city: Another City object
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1 = np.radians(self.latitude), np.radians(self.longitude)
        lat2, lon2 = np.radians(other_city.latitude), np.radians(other_city.longitude)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        R = 6371
        return R * c
    
    def get_demographic_summary(self) -> Dict[str, int]:
        """
        Get a summary of demographics as a dictionary.
        
        Returns:
            Dictionary with origin city names as keys and population counts as values
        """
        return {demo.origin_city: demo.population_count for demo in self.demographics}
    
    def update_year(self, new_year: int) -> None:
        """
        Update the year for the city data.
        
        Args:
            new_year: New year value
        """
        self.year = new_year
    
    def get_year(self) -> int:
        """
        Get the current year for the city data.
        
        Returns:
            Current year
        """
        return self.year
    
    def __repr__(self) -> str:
        """String representation of the City object."""
        return (f"City(name='{self.name}', "
                f"longitude={self.longitude}, "
                f"latitude={self.latitude}, "
                f"population={self.population}, "
                f"labor_immigrants={self.get_total_labor_immigrants()}, "
                f"year={self.year})")
    
    def __str__(self) -> str:
        """Detailed string representation of the City object."""
        lines = [
            f"City: {self.name}",
            f"Coordinates: ({self.latitude}, {self.longitude})"
        ]
        
        if self.demographics:
            lines.append("Labor Demographics:")
            for demo in self.demographics:
                lines.append(f"  - {demo.origin_city}: {demo.population_count:,}")
        
        lines.append(f"Cobb-Douglas Parameters: α={self.alpha:.3f}, "
                    f"β={self.beta:.3f}, A={self.A:.3f}")
        if self.capital_stock > 0 and self.labor_force > 0:
            production = self.calculate_production()
            gdp_per_capita = self.calculate_gdp_per_capita()
            lines.append(f"Production Output: {production:,.2f}")
            lines.append(f"GDP per Capita: {gdp_per_capita:,.2f}")
        
        return "\n".join(lines)

    def update_capital_stock(self, capital_stock: float) -> None:
        """
        Update the capital stock for production function.
        
        Args:
            capital_stock: New capital stock value
        """
        self.capital_stock = capital_stock
    
    def update_capital_stock_ratio(self, ratio: float) -> None:
        """
        Update the capital stock by a ratio (percentage change).
        
        Args:
            ratio: Ratio multiplier (e.g., 1.1 for 10% increase, 0.9 for 10% decrease)
        """
        self.capital_stock *= ratio
    
    def update_labor_force_ratio(self, ratio: float) -> None:
        """
        Update the labor force by a ratio (percentage change).
        
        Args:
            ratio: Ratio multiplier (e.g., 1.1 for 10% increase, 0.9 for 10% decrease)
        """
        self.labor_force = int(self.labor_force * ratio)
    
    def update_cobb_douglas_params(self, alpha_ratio: float = None, beta_ratio: float = None, A_ratio: float = None) -> None:
        """
        Update Cobb-Douglas production function parameters by ratios.
        
        Args:
            alpha_ratio: Ratio multiplier for capital share parameter (e.g., 1.1 for 10% increase)
            beta_ratio: Ratio multiplier for labor share parameter (e.g., 1.1 for 10% increase)
            A_ratio: Ratio multiplier for total factor productivity (e.g., 1.1 for 10% increase)
        """
        # Update existing parameters by ratio
        if alpha_ratio is not None:
            self.alpha *= alpha_ratio
        if beta_ratio is not None:
            self.beta *= beta_ratio
        if A_ratio is not None:
            self.A *= A_ratio
    
    def update_alpha(self, ratio: float) -> None:
        """
        Update the capital share parameter (alpha) by a ratio.
        
        Args:
            ratio: Ratio multiplier (e.g., 1.1 for 10% increase, 0.9 for 10% decrease)
        """
        self.alpha *= ratio
    
    def update_beta(self, ratio: float) -> None:
        """
        Update the labor share parameter (beta) by a ratio.
        
        Args:
            ratio: Ratio multiplier (e.g., 1.1 for 10% increase, 0.9 for 10% decrease)
        """
        self.beta *= ratio
    
    def update_total_factor_productivity(self, ratio: float) -> None:
        """
        Update the total factor productivity (A) by a ratio.
        
        Args:
            ratio: Ratio multiplier (e.g., 1.1 for 10% increase, 0.9 for 10% decrease)
        """
        self.A *= ratio
    
    def get_cobb_douglas_params(self) -> Dict[str, float]:
        """
        Get the current Cobb-Douglas parameters.
        
        Returns:
            Dictionary with alpha, beta, and A values
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'A': self.A
        }
    
    def get_economic_summary(self) -> Dict[str, float]:
        """
        Get a summary of economic parameters.
        
        Returns:
            Dictionary with economic parameters
        """
        summary = {
            'labor_force': self.labor_force,
            'capital_stock': self.capital_stock,
            'alpha': self.alpha,
            'beta': self.beta,
            'total_factor_productivity': self.A
        }
        
        if self.capital_stock > 0 and self.labor_force > 0:
            summary['production_output'] = self.calculate_production()
            summary['gdp_per_capita'] = self.calculate_gdp_per_capita()
        
        return summary

    def get_all_info(self) -> Dict[str, any]:
        """
        Get all information about the city as a dictionary for dataset creation.
        
        Returns:
            Dictionary containing all city information including:
            - Basic info (name, coordinates)
            - Demographics (total population, immigrants, demographic breakdown)
            - Economic parameters (Cobb-Douglas parameters, capital, labor)
            - Calculated values (production, GDP per capita)
            - Geographic info (coordinates)
        """
        info = {
            # Basic city information
            'city_name': self.name,
            'year': self.year,
            'longitude': self.longitude,
            'latitude': self.latitude,
            
            # Population and demographics
            'total_population': self.population,
            'total_labor_force': self.labor_force,
            'total_labor_immigrants': self.get_total_labor_immigrants(),
            'labor_native': self.get_labor_native(),
            'labor_immigration_rate': self.get_total_labor_immigrants() / self.labor_force if self.labor_force > 0 else 0,
            
            # Economic parameters
            'capital_stock': self.capital_stock,
            'alpha': self.alpha,
            'beta': self.beta,
            'total_factor_productivity': self.A,
            
            # Demographic breakdown by origin city
            'labor_demographic_breakdown': self.get_demographic_summary()
        }
        
        # Add calculated economic values if possible
        if self.capital_stock > 0 and self.labor_force > 0:
            info.update({
                'production_output': self.calculate_production(),
                'gdp_per_capita': self.calculate_gdp_per_capita(),
                'capital_labor_ratio': self.capital_stock / self.labor_force,
                'capital_per_capita': self.capital_stock / self.labor_force
            })
        else:
            info.update({
                'production_output': 0.0,
                'gdp_per_capita': 0.0,
                'capital_labor_ratio': 0.0,
                'capital_per_capita': 0.0
            })
        
        # Add individual demographic counts as separate fields
        for demo in self.demographics:
            field_name = f"immigrants_from_{demo.origin_city.lower().replace(' ', '_')}"
            info[field_name] = demo.population_count
        
        return info

    def migrate_labor_to(self, destination_city: 'City', ratio: float) -> None:
        """
        Migrate a ratio of labor force from this city to another city and update destination demographics.
        The migrated labor is subtracted proportionally from each demographic group in the source city.
        
        Args:
            destination_city: The City object to which labor will migrate
            ratio: The ratio of labor force to migrate (e.g., 0.1 for 10%)
        """
        if not (0 < ratio <= 1):
            raise ValueError("Migration ratio must be between 0 and 1.")
        if self.labor_force <= 0:
            raise ValueError("Source city has no labor force to migrate.")
        migrating_labor = int(self.labor_force * ratio)
        if migrating_labor < 1:
            migrating_labor = 1  # Ensure at least 1 person migrates if ratio > 0
        if migrating_labor > self.labor_force:
            migrating_labor = self.labor_force
        self.labor_force -= migrating_labor
        destination_city.labor_force += migrating_labor
        # Update destination city's demographics
        destination_city.add_labor_demographic(self.name, migrating_labor)
        # Proportionally decrease from each demographic group
        total_demo = sum(demo.population_count for demo in self.demographics)
        if total_demo > 0:
            removed = 0
            for i, demo in enumerate(self.demographics):
                if i == len(self.demographics) - 1:
                    # Last group gets the remainder
                    to_remove = migrating_labor - removed
                else:
                    to_remove = int(round(migrating_labor * (demo.population_count / total_demo)))
                    removed += to_remove
                demo.population_count -= min(to_remove, demo.population_count)
            # Remove any demographic group that drops to zero or below
            self.demographics = [d for d in self.demographics if d.population_count > 0]

    def migrate_capital_to(self, destination_city: 'City', ratio: float) -> None:
        """
        Migrate a ratio of capital stock from this city to another city.
        
        Args:
            destination_city: The City object to which capital will migrate
            ratio: The ratio of capital stock to migrate (e.g., 0.1 for 10%)
        """
        if not (0 < ratio <= 1):
            raise ValueError("Migration ratio must be between 0 and 1.")
        if self.capital_stock <= 0:
            raise ValueError("Source city has no capital stock to migrate.")
        migrating_capital = self.capital_stock * ratio
        if migrating_capital < 1:
            migrating_capital = 1  # Ensure at least 1 unit migrates if ratio > 0
        if migrating_capital > self.capital_stock:
            migrating_capital = self.capital_stock
        self.capital_stock -= migrating_capital
        destination_city.capital_stock += migrating_capital


# Example usage and testing
if __name__ == "__main__":
    # Create a city with Cobb-Douglas parameters directly
    istanbul = City(
        name="Istanbul",
        longitude=28.9784,
        latitude=41.0082,
        population=15000000,
        capital_stock=1000000,
        alpha=0.3,
        beta=0.7,
        A=1.5
    )
    
    # Add demographic information
    istanbul.add_labor_demographic("Ankara", 500000)
    istanbul.add_labor_demographic("Izmir", 300000)
    istanbul.add_labor_demographic("Bursa", 200000)
    
    # Create another city
    ankara = City(
        name="Ankara",
        longitude=32.8597,
        latitude=39.9334,
        population=5500000,
        capital_stock=500000
    )
    
    # Print city information
    print(istanbul)
    print("\n" + "="*50 + "\n")
    print(ankara)
    
    # Calculate distance between cities
    distance = istanbul.distance_to(ankara)
    print(f"\nDistance between Istanbul and Ankara: {distance:.1f} km")
    
    # Calculate production
    production = istanbul.calculate_production()
    print(f"Istanbul's production output: {production:,.2f}")
    
    # Calculate GDP per capita
    gdp_per_capita = istanbul.calculate_gdp_per_capita()
    print(f"Istanbul's GDP per capita: {gdp_per_capita:,.2f}")
    
    # Demonstrate update methods
    print("\n" + "="*50)
    print("DEMONSTRATING RATIO-BASED UPDATE METHODS")
    print("="*50)
    
    # Show initial values
    print(f"Initial alpha: {istanbul.alpha:.3f}")
    print(f"Initial beta: {istanbul.beta:.3f}")
    print(f"Initial A: {istanbul.A:.3f}")
    print(f"Initial labor force: {istanbul.labor_force:,}")
    print(f"Initial capital stock: {istanbul.capital_stock:,.0f}")
    
    # Update labor force by 10% increase
    istanbul.update_labor_force_ratio(1.1)
    print(f"After 10% labor force increase: {istanbul.labor_force:,}")
    
    # Update capital stock by 20% increase
    istanbul.update_capital_stock_ratio(1.2)
    print(f"After 20% capital stock increase: {istanbul.capital_stock:,.0f}")
    
    # Update Cobb-Douglas parameters by ratios
    istanbul.update_cobb_douglas_params(alpha_ratio=1.1, beta_ratio=0.95, A_ratio=1.15)
    print(f"After ratio updates: α={istanbul.alpha:.3f}, β={istanbul.beta:.3f}, A={istanbul.A:.3f}")
    
    # Update individual parameters by ratios
    istanbul.update_alpha(0.9)  # 10% decrease
    istanbul.update_beta(1.05)  # 5% increase
    istanbul.update_total_factor_productivity(1.1)  # 10% increase
    print(f"Individual ratio updates: α={istanbul.alpha:.3f}, β={istanbul.beta:.3f}, A={istanbul.A:.3f}")
    
    # Get economic summary
    economic_summary = istanbul.get_economic_summary()
    print("\nEconomic Summary:")
    for key, value in economic_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value:,}")
    
    # Calculate new production after updates
    new_production = istanbul.calculate_production()
    print(f"\nNew production output after ratio updates: {new_production:,.2f}")
    
    # Calculate new GDP per capita after updates
    new_gdp_per_capita = istanbul.calculate_gdp_per_capita()
    print(f"New GDP per capita after ratio updates: {new_gdp_per_capita:,.2f}")
    
    # Demonstrate get_all_info for dataset creation
    print("\n" + "="*50)
    print("DATASET CREATION EXAMPLE")
    print("="*50)
    
    # Get all information as dictionary for dataset row
    istanbul_data = istanbul.get_all_info()
    print("Istanbul city data for dataset:")
    for key, value in istanbul_data.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Example of creating multiple city rows for a dataset
    print("\nExample of multiple cities for dataset:")
    cities = [istanbul, ankara]
    dataset_rows = []
    
    for city in cities:
        city_data = city.get_all_info()
        dataset_rows.append(city_data)
        print(f"\n{city.name} row added to dataset with {len(city_data)} fields")
    
    print(f"\nTotal dataset rows: {len(dataset_rows)}")
    print(f"Dataset columns: {list(dataset_rows[0].keys()) if dataset_rows else 'No data'}") 