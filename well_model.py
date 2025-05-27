import numpy as np

class WellModel:
    """
    Well model for reservoir simulation
    
    Handles production and injection wells with various control modes
    (rate control, pressure control, etc.)
    """
    
    def __init__(self, reservoir):
        self.reservoir = reservoir
        self.wells = []
    
    def add_well(self, well_data):
        """
        Add a well to the model
        
        Args:
            well_data: Dictionary containing well information
                - name: Well name
                - type: 'producer' or 'injector'
                - i, j, k: Grid location
                - rate: Flow rate (positive for injection, negative for production)
                - bhp: Bottom hole pressure (optional)
                - radius: Well radius
        """
        # Set default values
        well = {
            'name': well_data.get('name', f'WELL_{len(self.wells)+1}'),
            'type': well_data.get('type', 'producer'),
            'i': well_data.get('i', 0),
            'j': well_data.get('j', 0),
            'k': well_data.get('k', 0),
            'rate': well_data.get('rate', 0.0),
            'bhp': well_data.get('bhp', None),
            'radius': well_data.get('radius', 0.1),  # meters
            'skin': well_data.get('skin', 0.0),
            'active': True
        }
        
        # Validate grid location
        well['i'] = max(0, min(well['i'], self.reservoir.nx - 1))
        well['j'] = max(0, min(well['j'], self.reservoir.ny - 1))
        well['k'] = max(0, min(well['k'], self.reservoir.nz - 1))
        
        self.wells.append(well)
        return well
    
    def calculate_well_index(self, well):
        """
        Calculate well index (productivity/injectivity index)
        
        Args:
            well: Well dictionary
        
        Returns:
            WI: Well index in m³/(Pa·s)
        """
        i, j, k = well['i'], well['j'], well['k']
        
        # Grid block dimensions
        dx = self.reservoir.dx
        dy = self.reservoir.dy
        dz = self.reservoir.dz
        
        # Permeability in well block
        kx = self.reservoir.permeability_x[i, j, k]
        ky = self.reservoir.permeability_y[i, j, k]
        
        # Equivalent radius (Peaceman's formula)
        r_eq = 0.28 * np.sqrt((ky/kx)**0.5 * dx**2 + (kx/ky)**0.5 * dy**2) / \
               ((ky/kx)**0.25 + (kx/ky)**0.25)
        
        # Geometric mean permeability
        k_avg = np.sqrt(kx * ky)
        
        # Well index
        well_index = (2 * np.pi * k_avg * dz) / \
                    (np.log(r_eq / well['radius']) + well['skin'])
        
        return well_index
    
    def calculate_well_rates(self):
        """
        Calculate actual flow rates for all wells based on pressure constraints
        
        Returns:
            Dictionary with well rates and phase splits
        """
        well_rates = {}
        
        for well in self.wells:
            if not well['active']:
                continue
            
            i, j, k = well['i'], well['j'], well['k']
            well_index = self.calculate_well_index(well)
            
            # Grid block pressure
            p_grid = self.reservoir.pressure[i, j, k]
            
            if well['type'] == 'producer':
                # Production well
                if well['bhp'] is not None:
                    # Pressure constraint
                    bhp = well['bhp']
                    pressure_diff = p_grid - bhp
                    
                    if pressure_diff > 0:
                        # Calculate phase rates based on mobility
                        total_rate = self._calculate_total_well_rate(well, pressure_diff, well_index)
                        phase_rates = self._split_well_rate_by_phases(well, total_rate)
                    else:
                        # Well cannot produce
                        phase_rates = {'oil': 0, 'gas': 0, 'water': 0, 'total': 0}
                else:
                    # Rate constraint
                    total_rate = abs(well['rate'])
                    phase_rates = self._split_well_rate_by_phases(well, total_rate)
            
            elif well['type'] == 'injector':
                # Injection well
                if well['bhp'] is not None:
                    # Maximum injection pressure constraint
                    max_bhp = well['bhp']
                    pressure_diff = max_bhp - p_grid
                    
                    if pressure_diff > 0:
                        total_rate = self._calculate_total_well_rate(well, pressure_diff, well_index)
                        # Injection is typically single phase (water or gas)
                        phase_rates = self._split_injection_rate(well, total_rate)
                    else:
                        # Cannot inject
                        phase_rates = {'oil': 0, 'gas': 0, 'water': 0, 'total': 0}
                else:
                    # Rate constraint
                    total_rate = well['rate']
                    phase_rates = self._split_injection_rate(well, total_rate)
            
            else:
                phase_rates = {'oil': 0, 'gas': 0, 'water': 0, 'total': 0}
            
            well_rates[well['name']] = phase_rates
        
        return well_rates
    
    def _calculate_total_well_rate(self, well, pressure_diff, well_index):
        """Calculate total well rate based on pressure difference and mobility"""
        i, j, k = well['i'], well['j'], well['k']
        
        # Calculate total mobility
        p = self.reservoir.pressure[i, j, k]
        
        # Oil mobility
        kro = self.reservoir.get_relative_permeability(
            self.reservoir.oil_saturation[i, j, k], 'oil')
        mu_o = self.reservoir.pvt.oil_viscosity(p)
        lambda_o = kro / mu_o
        
        # Gas mobility
        krg = self.reservoir.get_relative_permeability(
            self.reservoir.gas_saturation[i, j, k], 'gas')
        mu_g = self.reservoir.pvt.gas_viscosity(p)
        lambda_g = krg / mu_g
        
        # Water mobility
        krw = self.reservoir.get_relative_permeability(
            self.reservoir.water_saturation[i, j, k], 'water')
        mu_w = self.reservoir.pvt.water_viscosity(p)
        lambda_w = krw / mu_w
        
        total_mobility = lambda_o + lambda_g + lambda_w
        
        # Total rate
        total_rate = well_index * total_mobility * pressure_diff
        
        return total_rate
    
    def _split_well_rate_by_phases(self, well, total_rate):
        """Split total well rate into phase rates based on fractional flow"""
        i, j, k = well['i'], well['j'], well['k']
        p = self.reservoir.pressure[i, j, k]
        
        # Calculate mobilities
        kro = self.reservoir.get_relative_permeability(
            self.reservoir.oil_saturation[i, j, k], 'oil')
        mu_o = self.reservoir.pvt.oil_viscosity(p)
        lambda_o = kro / mu_o
        
        krg = self.reservoir.get_relative_permeability(
            self.reservoir.gas_saturation[i, j, k], 'gas')
        mu_g = self.reservoir.pvt.gas_viscosity(p)
        lambda_g = krg / mu_g
        
        krw = self.reservoir.get_relative_permeability(
            self.reservoir.water_saturation[i, j, k], 'water')
        mu_w = self.reservoir.pvt.water_viscosity(p)
        lambda_w = krw / mu_w
        
        total_mobility = lambda_o + lambda_g + lambda_w
        
        if total_mobility > 0:
            # Fractional flows
            f_o = lambda_o / total_mobility
            f_g = lambda_g / total_mobility
            f_w = lambda_w / total_mobility
            
            # Phase rates
            oil_rate = total_rate * f_o
            gas_rate = total_rate * f_g
            water_rate = total_rate * f_w
        else:
            oil_rate = gas_rate = water_rate = 0
        
        return {
            'oil': oil_rate,
            'gas': gas_rate,
            'water': water_rate,
            'total': total_rate
        }
    
    def _split_injection_rate(self, well, total_rate):
        """Split injection rate (typically single phase)"""
        # Assume water injection by default
        injection_phase = getattr(well, 'injection_phase', 'water')
        
        phase_rates = {'oil': 0, 'gas': 0, 'water': 0, 'total': total_rate}
        
        if injection_phase == 'water':
            phase_rates['water'] = total_rate
        elif injection_phase == 'gas':
            phase_rates['gas'] = total_rate
        else:
            # Default to water
            phase_rates['water'] = total_rate
        
        return phase_rates
    
    def apply_well_source_terms(self, source_terms):
        """
        Apply well source terms to the grid
        
        Args:
            source_terms: Dictionary to store source terms for each phase
        """
        well_rates = self.calculate_well_rates()
        
        for well in self.wells:
            if not well['active']:
                continue
            
            well_name = well['name']
            if well_name not in well_rates:
                continue
            
            i, j, k = well['i'], well['j'], well['k']
            rates = well_rates[well_name]
            
            # Add source terms (positive for injection, negative for production)
            if well['type'] == 'producer':
                source_terms['oil'][i, j, k] -= rates['oil']
                source_terms['gas'][i, j, k] -= rates['gas']
                source_terms['water'][i, j, k] -= rates['water']
            elif well['type'] == 'injector':
                source_terms['oil'][i, j, k] += rates['oil']
                source_terms['gas'][i, j, k] += rates['gas']
                source_terms['water'][i, j, k] += rates['water']
    
    def get_well_summary(self):
        """Get summary of all wells"""
        summary = []
        
        for well in self.wells:
            well_info = {
                'name': well['name'],
                'type': well['type'],
                'location': f"({well['i']}, {well['j']}, {well['k']})",
                'rate': well['rate'],
                'active': well['active']
            }
            
            if well['bhp'] is not None:
                well_info['bhp'] = well['bhp']
            
            summary.append(well_info)
        
        return summary
    
    def update_well_controls(self, well_name, **kwargs):
        """Update well control parameters"""
        for well in self.wells:
            if well['name'] == well_name:
                for key, value in kwargs.items():
                    if key in well:
                        well[key] = value
                break
    
    def shut_in_well(self, well_name):
        """Shut in a well"""
        for well in self.wells:
            if well['name'] == well_name:
                well['active'] = False
                break
    
    def open_well(self, well_name):
        """Open a well"""
        for well in self.wells:
            if well['name'] == well_name:
                well['active'] = True
                break
