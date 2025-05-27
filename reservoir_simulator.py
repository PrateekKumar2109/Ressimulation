import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

from pvt_properties import PVTProperties
from numerical_solver import NumericalSolver
from well_model import WellModel

class ReservoirSimulator:
    """
    3D Black Oil Reservoir Simulator
    
    Implements finite difference solution of multiphase flow equations
    for oil, gas, and water in a 3D grid system.
    """
    
    def __init__(self, nx=20, ny=20, nz=5, 
                 length_x=1000, length_y=1000, length_z=50,
                 porosity=0.2, permeability_x=100, permeability_y=100, permeability_z=10,
                 initial_pressure=3000, initial_oil_saturation=0.7, initial_water_saturation=0.3):
        
        # Grid dimensions
        self.nx, self.ny, self.nz = nx, ny, nz
        self.total_cells = nx * ny * nz
        
        # Physical dimensions
        self.length_x = length_x
        self.length_y = length_y
        self.length_z = length_z
        
        # Grid spacing
        self.dx = length_x / nx
        self.dy = length_y / ny
        self.dz = length_z / nz
        
        # Rock properties
        self.porosity = np.full((nx, ny, nz), porosity)
        self.permeability_x = np.full((nx, ny, nz), permeability_x * 9.869233e-16)  # Convert mD to m²
        self.permeability_y = np.full((nx, ny, nz), permeability_y * 9.869233e-16)
        self.permeability_z = np.full((nx, ny, nz), permeability_z * 9.869233e-16)
        
        # Initialize pressure and saturations
        self.pressure = np.full((nx, ny, nz), initial_pressure * 6895)  # Convert psia to Pa
        self.oil_saturation = np.full((nx, ny, nz), initial_oil_saturation)
        self.water_saturation = np.full((nx, ny, nz), initial_water_saturation)
        self.gas_saturation = 1.0 - self.oil_saturation - self.water_saturation
        
        # Ensure saturation constraint
        self.gas_saturation = np.maximum(self.gas_saturation, 0.0)
        
        # Previous time step values
        self.pressure_old = self.pressure.copy()
        self.oil_saturation_old = self.oil_saturation.copy()
        self.water_saturation_old = self.water_saturation.copy()
        self.gas_saturation_old = self.gas_saturation.copy()
        
        # Initialize PVT properties
        self.pvt = PVTProperties()
        
        # Initialize numerical solver
        self.solver = NumericalSolver(self)
        
        # Well model
        self.well_model = WellModel(self)
        self.wells = []
        
        # Simulation time
        self.current_time = 0.0
        self.time_step = 1.0
        
        # History storage
        self.production_history = []
        self.material_balance_history = []
        
        # Calculate initial material balance
        self._update_material_balance()
    
    def add_wells(self, num_producers=2, num_injectors=1, production_rate=500, injection_rate=300):
        """Add production and injection wells to the reservoir"""
        self.wells = []
        
        # Add producers
        for i in range(num_producers):
            # Distribute producers around the reservoir
            x_pos = int(self.nx * 0.25 + i * self.nx * 0.5)
            y_pos = int(self.ny * 0.25 + i * self.ny * 0.5)
            z_pos = int(self.nz / 2)
            
            well = {
                'name': f'PROD_{i+1}',
                'type': 'producer',
                'i': min(x_pos, self.nx-1),
                'j': min(y_pos, self.ny-1),
                'k': z_pos,
                'rate': -production_rate * 0.159  # Convert STB/day to m³/day
            }
            self.wells.append(well)
        
        # Add injectors
        for i in range(num_injectors):
            # Place injectors at reservoir edges
            x_pos = int(self.nx * 0.1) if i % 2 == 0 else int(self.nx * 0.9)
            y_pos = int(self.ny * 0.5)
            z_pos = int(self.nz / 2)
            
            well = {
                'name': f'INJ_{i+1}',
                'type': 'injector',
                'i': min(x_pos, self.nx-1),
                'j': min(y_pos, self.ny-1),
                'k': z_pos,
                'rate': injection_rate * 0.159  # Convert STB/day to m³/day
            }
            self.wells.append(well)
    
    def get_relative_permeability(self, saturation, phase):
        """Calculate relative permeability using Corey correlations"""
        if phase == 'oil':
            # Oil relative permeability
            s_norm = (saturation - 0.2) / (0.8 - 0.2)  # Normalize between connate and residual
            s_norm = np.clip(s_norm, 0, 1)
            return s_norm ** 2
        
        elif phase == 'water':
            # Water relative permeability
            s_norm = (saturation - 0.2) / (0.8 - 0.2)
            s_norm = np.clip(s_norm, 0, 1)
            return s_norm ** 2
        
        elif phase == 'gas':
            # Gas relative permeability
            s_norm = (saturation - 0.05) / (0.95 - 0.05)
            s_norm = np.clip(s_norm, 0, 1)
            return s_norm ** 2
        
        return np.zeros_like(saturation)
    
    def step(self, dt, max_dt=30):
        """Advance simulation by one time step"""
        try:
            # Adaptive time stepping
            dt = min(dt, max_dt)
            
            # Store old values
            self.pressure_old = self.pressure.copy()
            self.oil_saturation_old = self.oil_saturation.copy()
            self.water_saturation_old = self.water_saturation.copy()
            self.gas_saturation_old = self.gas_saturation.copy()
            
            # Solve pressure equation
            self.solver.solve_pressure(dt)
            
            # Solve saturation equations
            self.solver.solve_saturations(dt)
            
            # Update saturations to maintain constraint
            self._enforce_saturation_constraint()
            
            # Update time
            self.current_time += dt
            self.time_step = dt
            
            # Calculate production rates
            self._calculate_production()
            
            # Update material balance
            self._update_material_balance()
            
        except Exception as e:
            print(f"Simulation step error: {e}")
            # Revert to previous time step values
            self.pressure = self.pressure_old.copy()
            self.oil_saturation = self.oil_saturation_old.copy()
            self.water_saturation = self.water_saturation_old.copy()
            self.gas_saturation = self.gas_saturation_old.copy()
            raise e
    
    def _enforce_saturation_constraint(self):
        """Ensure saturation constraint So + Sg + Sw = 1"""
        total_sat = self.oil_saturation + self.gas_saturation + self.water_saturation
        
        # Normalize saturations
        self.oil_saturation /= total_sat
        self.gas_saturation /= total_sat
        self.water_saturation /= total_sat
        
        # Apply physical constraints
        self.oil_saturation = np.clip(self.oil_saturation, 0.01, 0.99)
        self.gas_saturation = np.clip(self.gas_saturation, 0.0, 0.99)
        self.water_saturation = np.clip(self.water_saturation, 0.01, 0.99)
        
        # Renormalize
        total_sat = self.oil_saturation + self.gas_saturation + self.water_saturation
        self.oil_saturation /= total_sat
        self.gas_saturation /= total_sat
        self.water_saturation /= total_sat
    
    def _calculate_production(self):
        """Calculate production rates for each well"""
        production_data = {
            'time': self.current_time,
            'oil_rate': 0,
            'gas_rate': 0,
            'water_rate': 0
        }
        
        for well in self.wells:
            i, j, k = well['i'], well['j'], well['k']
            
            if well['type'] == 'producer':
                # Calculate phase production based on mobility and saturation
                oil_mobility = self.get_relative_permeability(self.oil_saturation[i, j, k], 'oil')
                gas_mobility = self.get_relative_permeability(self.gas_saturation[i, j, k], 'gas') 
                water_mobility = self.get_relative_permeability(self.water_saturation[i, j, k], 'water')
                
                total_mobility = oil_mobility + gas_mobility + water_mobility
                
                if total_mobility > 0:
                    oil_frac = oil_mobility / total_mobility
                    gas_frac = gas_mobility / total_mobility
                    water_frac = water_mobility / total_mobility
                    
                    # Convert back to STB/day
                    oil_prod = abs(well['rate']) * oil_frac / 0.159
                    gas_prod = abs(well['rate']) * gas_frac / 0.159 * 1000  # Convert to SCF
                    water_prod = abs(well['rate']) * water_frac / 0.159
                    
                    production_data['oil_rate'] += oil_prod
                    production_data['gas_rate'] += gas_prod
                    production_data['water_rate'] += water_prod
        
        self.production_history.append(production_data)
    
    def _update_material_balance(self):
        """Calculate material balance"""
        # Calculate pore volume
        pore_volume = self.porosity * self.dx * self.dy * self.dz
        
        # Calculate oil, gas, and water in place
        oil_in_place = np.sum(self.oil_saturation * pore_volume) / 0.159  # Convert to STB
        gas_in_place = np.sum(self.gas_saturation * pore_volume) / 0.159 * 1000  # Convert to SCF
        water_in_place = np.sum(self.water_saturation * pore_volume) / 0.159  # Convert to STB
        
        balance_data = {
            'time': self.current_time,
            'oil_in_place': oil_in_place,
            'gas_in_place': gas_in_place,
            'water_in_place': water_in_place,
            'average_pressure': np.mean(self.pressure) / 6895  # Convert to psia
        }
        
        self.material_balance_history.append(balance_data)
    
    def reset(self):
        """Reset simulation to initial conditions"""
        self.current_time = 0.0
        self.pressure = np.full((self.nx, self.ny, self.nz), 3000 * 6895)
        self.oil_saturation = np.full((self.nx, self.ny, self.nz), 0.7)
        self.water_saturation = np.full((self.nx, self.ny, self.nz), 0.3)
        self.gas_saturation = 1.0 - self.oil_saturation - self.water_saturation
        
        self.production_history = []
        self.material_balance_history = []
        self._update_material_balance()
    
    def get_grid_coordinates(self):
        """Get grid coordinates for visualization"""
        x = np.linspace(0, self.length_x, self.nx)
        y = np.linspace(0, self.length_y, self.ny)
        z = np.linspace(0, self.length_z, self.nz)
        return x, y, z
