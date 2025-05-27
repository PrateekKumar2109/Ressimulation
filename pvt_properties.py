import numpy as np

class PVTProperties:
    """
    PVT (Pressure-Volume-Temperature) properties for black oil simulation
    
    Implements correlations for formation volume factors, viscosities,
    and densities for oil, gas, and water phases.
    """
    
    def __init__(self):
        # Reference conditions (standard conditions)
        self.p_std = 14.7 * 6895  # Standard pressure in Pa (14.7 psia)
        self.t_std = 288.15  # Standard temperature in K (60°F)
        
        # Oil properties at standard conditions
        self.oil_density_std = 850  # kg/m³ (API ~35)
        self.oil_viscosity_std = 1e-3  # Pa·s (1 cP)
        
        # Gas properties at standard conditions
        self.gas_density_std = 0.8  # kg/m³
        self.gas_viscosity_std = 1.8e-5  # Pa·s
        
        # Water properties at standard conditions
        self.water_density_std = 1000  # kg/m³
        self.water_viscosity_std = 1e-3  # Pa·s (1 cP)
        
        # Compressibilities
        self.oil_compressibility = 1e-9  # Pa⁻¹ (typical value ~1e-6 psi⁻¹)
        self.gas_compressibility = 1e-6  # Pa⁻¹
        self.water_compressibility = 4.5e-10  # Pa⁻¹
    
    def oil_formation_volume_factor(self, pressure, temperature=350):
        """
        Calculate oil formation volume factor (Bo)
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            Bo: Oil formation volume factor (reservoir bbl/STB)
        """
        # Convert pressure to psia for correlation
        p_psia = pressure / 6895
        
        # Simple correlation for undersaturated oil
        # Bo = Bo_initial * exp(-co * (p - p_initial))
        p_initial = 3000  # psia
        bo_initial = 1.2  # bbl/STB
        
        # Oil compressibility effect
        co = 1e-6  # psi⁻¹
        bo = bo_initial * np.exp(-co * (p_psia - p_initial))
        
        return np.maximum(bo, 1.0)  # Bo should be >= 1.0
    
    def gas_formation_volume_factor(self, pressure, temperature=350):
        """
        Calculate gas formation volume factor (Bg)
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            Bg: Gas formation volume factor (reservoir bbl/SCF)
        """
        # Convert pressure to psia
        p_psia = pressure / 6895
        t_rankine = temperature * 1.8  # Convert K to °R
        
        # Gas law: Bg = (z * T) / (p * 5.615)
        # Assuming z ≈ 1 for simplicity
        z_factor = 1.0
        bg = (z_factor * t_rankine) / (p_psia * 5.615)
        
        return bg * 1e-3  # Convert to reasonable units
    
    def water_formation_volume_factor(self, pressure, temperature=350):
        """
        Calculate water formation volume factor (Bw)
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            Bw: Water formation volume factor (reservoir bbl/STB)
        """
        # Water is slightly compressible
        # Bw = Bw_std * exp(-cw * (p - p_std))
        p_psia = pressure / 6895
        p_std_psia = 14.7
        
        bw_std = 1.0
        cw = 3e-6  # psi⁻¹
        
        bw = bw_std * np.exp(-cw * (p_psia - p_std_psia))
        
        return bw
    
    def oil_viscosity(self, pressure, temperature=350):
        """
        Calculate oil viscosity
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            μo: Oil viscosity in Pa·s
        """
        # Temperature effect (Arrhenius-type)
        t_ref = 350  # K
        activation_energy = 2000  # J/mol
        r_gas = 8.314  # J/(mol·K)
        
        mu_temp = self.oil_viscosity_std * np.exp(activation_energy / r_gas * (1/temperature - 1/t_ref))
        
        # Pressure effect (slight increase with pressure)
        p_psia = pressure / 6895
        mu_pressure_factor = 1 + 1e-6 * p_psia
        
        return mu_temp * mu_pressure_factor
    
    def gas_viscosity(self, pressure, temperature=350):
        """
        Calculate gas viscosity
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            μg: Gas viscosity in Pa·s
        """
        # Simple correlation for natural gas
        # μg ∝ T^0.5 / (molecular weight)
        t_ratio = (temperature / self.t_std) ** 0.5
        mu_gas = self.gas_viscosity_std * t_ratio
        
        return mu_gas
    
    def water_viscosity(self, pressure, temperature=350):
        """
        Calculate water viscosity
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            μw: Water viscosity in Pa·s
        """
        # Temperature-dependent viscosity for water
        # Simplified correlation
        if temperature > 273.15:
            mu_water = 2.414e-5 * 10**(247.8 / (temperature - 140))
        else:
            mu_water = self.water_viscosity_std
        
        return mu_water
    
    def oil_density(self, pressure, temperature=350):
        """
        Calculate oil density
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            ρo: Oil density in kg/m³
        """
        # Density changes with pressure and temperature
        bo = self.oil_formation_volume_factor(pressure, temperature)
        density = self.oil_density_std / bo
        
        return density
    
    def gas_density(self, pressure, temperature=350):
        """
        Calculate gas density
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            ρg: Gas density in kg/m³
        """
        # Ideal gas law: ρ = pM/(RT)
        molecular_weight = 0.02  # kg/mol (typical for natural gas)
        r_gas = 8.314  # J/(mol·K)
        
        density = (pressure * molecular_weight) / (r_gas * temperature)
        
        return density
    
    def water_density(self, pressure, temperature=350):
        """
        Calculate water density
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            ρw: Water density in kg/m³
        """
        # Water density with slight compressibility
        bw = self.water_formation_volume_factor(pressure, temperature)
        density = self.water_density_std / bw
        
        return density
    
    def solution_gas_oil_ratio(self, pressure, temperature=350):
        """
        Calculate solution gas-oil ratio (Rs)
        
        Args:
            pressure: Pressure in Pa
            temperature: Temperature in K
        
        Returns:
            Rs: Solution GOR in SCF/STB
        """
        # Simplified correlation for Rs
        p_psia = pressure / 6895
        
        # Below bubble point pressure
        if p_psia < 2000:  # Assume bubble point at 2000 psia
            rs = 100 * (p_psia / 1000) ** 1.2
        else:
            rs = 200  # Constant above bubble point
        
        return np.maximum(rs, 0)
