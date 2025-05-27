import numpy as np
from scipy.sparse import diags, csr_matrix, identity
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class NumericalSolver:
    """
    Numerical solver for multiphase flow equations using finite differences
    
    Implements IMPES (Implicit Pressure, Explicit Saturation) method
    for solving the black oil reservoir simulation equations.
    """
    
    def __init__(self, reservoir):
        self.reservoir = reservoir
        self.nx = reservoir.nx
        self.ny = reservoir.ny
        self.nz = reservoir.nz
        self.total_cells = reservoir.total_cells
        
        # Create index mapping for 3D to 1D conversion
        self._create_index_mapping()
        
        # Precompute geometric factors
        self._compute_geometric_factors()
    
    def _create_index_mapping(self):
        """Create mapping between 3D (i,j,k) and 1D indices"""
        self.index_map = np.zeros((self.nx, self.ny, self.nz), dtype=int)
        idx = 0
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    self.index_map[i, j, k] = idx
                    idx += 1
    
    def _compute_geometric_factors(self):
        """Precompute geometric factors for transmissibility calculations"""
        # Transmissibility factors
        self.trans_x = np.zeros((self.nx, self.ny, self.nz))
        self.trans_y = np.zeros((self.nx, self.ny, self.nz))
        self.trans_z = np.zeros((self.nx, self.ny, self.nz))
        
        # X-direction transmissibility
        for i in range(self.nx-1):
            for j in range(self.ny):
                for k in range(self.nz):
                    # Harmonic average of permeabilities
                    k1 = self.reservoir.permeability_x[i, j, k]
                    k2 = self.reservoir.permeability_x[i+1, j, k]
                    k_avg = 2 * k1 * k2 / (k1 + k2) if (k1 + k2) > 0 else 0
                    
                    # Area and distance
                    area = self.reservoir.dy * self.reservoir.dz
                    distance = self.reservoir.dx
                    
                    self.trans_x[i, j, k] = k_avg * area / distance
        
        # Y-direction transmissibility
        for i in range(self.nx):
            for j in range(self.ny-1):
                for k in range(self.nz):
                    k1 = self.reservoir.permeability_y[i, j, k]
                    k2 = self.reservoir.permeability_y[i, j+1, k]
                    k_avg = 2 * k1 * k2 / (k1 + k2) if (k1 + k2) > 0 else 0
                    
                    area = self.reservoir.dx * self.reservoir.dz
                    distance = self.reservoir.dy
                    
                    self.trans_y[i, j, k] = k_avg * area / distance
        
        # Z-direction transmissibility
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz-1):
                    k1 = self.reservoir.permeability_z[i, j, k]
                    k2 = self.reservoir.permeability_z[i, j, k+1]
                    k_avg = 2 * k1 * k2 / (k1 + k2) if (k1 + k2) > 0 else 0
                    
                    area = self.reservoir.dx * self.reservoir.dy
                    distance = self.reservoir.dz
                    
                    self.trans_z[i, j, k] = k_avg * area / distance
    
    def solve_pressure(self, dt):
        """
        Solve pressure equation implicitly
        
        ∇·(λt∇P) = ∂/∂t[(φct)P] + qt
        
        where λt is total mobility and ct is total compressibility
        """
        # Calculate total mobility and compressibility
        total_mobility = self._calculate_total_mobility()
        total_compressibility = self._calculate_total_compressibility()
        
        # Build coefficient matrix A and right-hand side b
        A, b = self._build_pressure_system(total_mobility, total_compressibility, dt)
        
        # Add well terms
        self._add_well_terms_pressure(A, b)
        
        # Solve linear system
        try:
            pressure_1d = spsolve(A, b)
            
            # Convert back to 3D
            idx = 0
            for k in range(self.nz):
                for j in range(self.ny):
                    for i in range(self.nx):
                        self.reservoir.pressure[i, j, k] = pressure_1d[idx]
                        idx += 1
                        
        except Exception as e:
            print(f"Pressure solver failed: {e}")
            # Use explicit update as fallback
            self._explicit_pressure_update(dt)
    
    def _calculate_total_mobility(self):
        """Calculate total mobility λt = λo + λg + λw"""
        total_mobility = np.zeros((self.nx, self.ny, self.nz))
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
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
                    
                    total_mobility[i, j, k] = lambda_o + lambda_g + lambda_w
        
        return total_mobility
    
    def _calculate_total_compressibility(self):
        """Calculate total compressibility ct"""
        # Simplified total compressibility
        # ct = cf + So*co + Sg*cg + Sw*cw
        formation_compressibility = 1e-10  # Pa⁻¹
        
        ct = (formation_compressibility + 
              self.reservoir.oil_saturation * self.reservoir.pvt.oil_compressibility +
              self.reservoir.gas_saturation * self.reservoir.pvt.gas_compressibility +
              self.reservoir.water_saturation * self.reservoir.pvt.water_compressibility)
        
        return ct
    
    def _build_pressure_system(self, total_mobility, total_compressibility, dt):
        """Build coefficient matrix and RHS for pressure equation"""
        n = self.total_cells
        
        # Initialize sparse matrix components
        row_indices = []
        col_indices = []
        data = []
        b = np.zeros(n)
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    idx = self.index_map[i, j, k]
                    
                    # Diagonal accumulation term
                    pore_volume = (self.reservoir.porosity[i, j, k] * 
                                 self.reservoir.dx * self.reservoir.dy * self.reservoir.dz)
                    accumulation = pore_volume * total_compressibility[i, j, k] / dt
                    
                    diagonal = accumulation
                    
                    # X-direction connections
                    if i > 0:
                        # Connection to left neighbor
                        k_avg = self.trans_x[i-1, j, k]
                        mobility_avg = 0.5 * (total_mobility[i, j, k] + total_mobility[i-1, j, k])
                        transmissibility = k_avg * mobility_avg
                        
                        neighbor_idx = self.index_map[i-1, j, k]
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(transmissibility)
                        diagonal += transmissibility
                    
                    if i < self.nx - 1:
                        # Connection to right neighbor
                        k_avg = self.trans_x[i, j, k]
                        mobility_avg = 0.5 * (total_mobility[i, j, k] + total_mobility[i+1, j, k])
                        transmissibility = k_avg * mobility_avg
                        
                        neighbor_idx = self.index_map[i+1, j, k]
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(transmissibility)
                        diagonal += transmissibility
                    
                    # Y-direction connections
                    if j > 0:
                        k_avg = self.trans_y[i, j-1, k]
                        mobility_avg = 0.5 * (total_mobility[i, j, k] + total_mobility[i, j-1, k])
                        transmissibility = k_avg * mobility_avg
                        
                        neighbor_idx = self.index_map[i, j-1, k]
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(transmissibility)
                        diagonal += transmissibility
                    
                    if j < self.ny - 1:
                        k_avg = self.trans_y[i, j, k]
                        mobility_avg = 0.5 * (total_mobility[i, j, k] + total_mobility[i, j+1, k])
                        transmissibility = k_avg * mobility_avg
                        
                        neighbor_idx = self.index_map[i, j+1, k]
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(transmissibility)
                        diagonal += transmissibility
                    
                    # Z-direction connections
                    if k > 0:
                        k_avg = self.trans_z[i, j, k-1]
                        mobility_avg = 0.5 * (total_mobility[i, j, k] + total_mobility[i, j, k-1])
                        transmissibility = k_avg * mobility_avg
                        
                        neighbor_idx = self.index_map[i, j, k-1]
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(transmissibility)
                        diagonal += transmissibility
                    
                    if k < self.nz - 1:
                        k_avg = self.trans_z[i, j, k]
                        mobility_avg = 0.5 * (total_mobility[i, j, k] + total_mobility[i, j, k+1])
                        transmissibility = k_avg * mobility_avg
                        
                        neighbor_idx = self.index_map[i, j, k+1]
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(transmissibility)
                        diagonal += transmissibility
                    
                    # Add diagonal term
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(-diagonal)
                    
                    # Right-hand side (accumulation term)
                    b[idx] = -accumulation * self.reservoir.pressure_old[i, j, k]
        
        # Create sparse matrix
        A = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        
        return A, b
    
    def _add_well_terms_pressure(self, A, b):
        """Add well terms to pressure equation"""
        for well in self.reservoir.wells:
            i, j, k = well['i'], well['j'], well['k']
            idx = self.index_map[i, j, k]
            
            # Well index (simplified)
            well_radius = 0.1  # m
            grid_radius = 0.5 * np.sqrt(self.reservoir.dx**2 + self.reservoir.dy**2)
            well_index = (2 * np.pi * self.reservoir.permeability_x[i, j, k] * self.reservoir.dz / 
                         np.log(grid_radius / well_radius))
            
            if well['type'] == 'producer':
                # For producers, use flowing bottom hole pressure constraint
                bhp = 1000 * 6895  # 1000 psia converted to Pa
                A[idx, idx] -= well_index
                b[idx] -= well_index * bhp
            
            elif well['type'] == 'injector':
                # For injectors, add rate directly to RHS
                b[idx] += well['rate'] * 86400  # Convert to m³/s to m³/day
    
    def _explicit_pressure_update(self, dt):
        """Fallback explicit pressure update"""
        # Simple explicit update for pressure
        pressure_change = np.zeros_like(self.reservoir.pressure)
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    # Simple diffusion update
                    d2p_dx2 = (self.reservoir.pressure[i+1, j, k] - 
                              2*self.reservoir.pressure[i, j, k] + 
                              self.reservoir.pressure[i-1, j, k]) / self.reservoir.dx**2
                    
                    d2p_dy2 = (self.reservoir.pressure[i, j+1, k] - 
                              2*self.reservoir.pressure[i, j, k] + 
                              self.reservoir.pressure[i, j-1, k]) / self.reservoir.dy**2
                    
                    d2p_dz2 = (self.reservoir.pressure[i, j, k+1] - 
                              2*self.reservoir.pressure[i, j, k] + 
                              self.reservoir.pressure[i, j, k-1]) / self.reservoir.dz**2
                    
                    diffusivity = 1e-6  # m²/s
                    pressure_change[i, j, k] = diffusivity * dt * (d2p_dx2 + d2p_dy2 + d2p_dz2)
        
        self.reservoir.pressure += pressure_change
    
    def solve_saturations(self, dt):
        """
        Solve saturation equations explicitly
        
        ∂S/∂t + ∇·(f_α v_t) = q_α/(ρ_α φ)
        
        where f_α is fractional flow and v_t is total velocity
        """
        # Calculate total velocity field
        total_velocity_x, total_velocity_y, total_velocity_z = self._calculate_total_velocity()
        
        # Update saturations using upwind scheme
        self._update_saturations_upwind(total_velocity_x, total_velocity_y, total_velocity_z, dt)
    
    def _calculate_total_velocity(self):
        """Calculate total velocity field from pressure gradient"""
        vx = np.zeros((self.nx, self.ny, self.nz))
        vy = np.zeros((self.nx, self.ny, self.nz))
        vz = np.zeros((self.nx, self.ny, self.nz))
        
        # X-direction velocity
        for i in range(self.nx-1):
            for j in range(self.ny):
                for k in range(self.nz):
                    dp_dx = (self.reservoir.pressure[i+1, j, k] - 
                            self.reservoir.pressure[i, j, k]) / self.reservoir.dx
                    
                    # Average properties
                    k_avg = 0.5 * (self.reservoir.permeability_x[i, j, k] + 
                                  self.reservoir.permeability_x[i+1, j, k])
                    
                    # Total mobility (simplified)
                    mobility = 1e-12  # m²/(Pa·s) - simplified average
                    
                    vx[i, j, k] = -k_avg * mobility * dp_dx
        
        # Y-direction velocity
        for i in range(self.nx):
            for j in range(self.ny-1):
                for k in range(self.nz):
                    dp_dy = (self.reservoir.pressure[i, j+1, k] - 
                            self.reservoir.pressure[i, j, k]) / self.reservoir.dy
                    
                    k_avg = 0.5 * (self.reservoir.permeability_y[i, j, k] + 
                                  self.reservoir.permeability_y[i, j+1, k])
                    
                    mobility = 1e-12
                    
                    vy[i, j, k] = -k_avg * mobility * dp_dy
        
        # Z-direction velocity
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz-1):
                    dp_dz = (self.reservoir.pressure[i, j, k+1] - 
                            self.reservoir.pressure[i, j, k]) / self.reservoir.dz
                    
                    k_avg = 0.5 * (self.reservoir.permeability_z[i, j, k] + 
                                  self.reservoir.permeability_z[i, j, k+1])
                    
                    mobility = 1e-12
                    
                    vz[i, j, k] = -k_avg * mobility * dp_dz
        
        return vx, vy, vz
    
    def _update_saturations_upwind(self, vx, vy, vz, dt):
        """Update saturations using upwind finite difference scheme"""
        # Calculate fractional flows
        oil_frac_flow = self._calculate_fractional_flow('oil')
        gas_frac_flow = self._calculate_fractional_flow('gas')
        water_frac_flow = self._calculate_fractional_flow('water')
        
        # Update oil saturation
        oil_sat_new = self.reservoir.oil_saturation.copy()
        
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                for k in range(1, self.nz-1):
                    # X-direction flux
                    if vx[i, j, k] > 0:
                        flux_x = vx[i, j, k] * oil_frac_flow[i, j, k]
                    else:
                        flux_x = vx[i, j, k] * oil_frac_flow[i+1, j, k]
                    
                    if vx[i-1, j, k] > 0:
                        flux_x_prev = vx[i-1, j, k] * oil_frac_flow[i-1, j, k]
                    else:
                        flux_x_prev = vx[i-1, j, k] * oil_frac_flow[i, j, k]
                    
                    # Y-direction flux
                    if vy[i, j, k] > 0:
                        flux_y = vy[i, j, k] * oil_frac_flow[i, j, k]
                    else:
                        flux_y = vy[i, j, k] * oil_frac_flow[i, j+1, k]
                    
                    if vy[i, j-1, k] > 0:
                        flux_y_prev = vy[i, j-1, k] * oil_frac_flow[i, j-1, k]
                    else:
                        flux_y_prev = vy[i, j-1, k] * oil_frac_flow[i, j, k]
                    
                    # Z-direction flux
                    if vz[i, j, k] > 0:
                        flux_z = vz[i, j, k] * oil_frac_flow[i, j, k]
                    else:
                        flux_z = vz[i, j, k] * oil_frac_flow[i, j, k+1]
                    
                    if k > 0 and vz[i, j, k-1] > 0:
                        flux_z_prev = vz[i, j, k-1] * oil_frac_flow[i, j, k-1]
                    elif k > 0:
                        flux_z_prev = vz[i, j, k-1] * oil_frac_flow[i, j, k]
                    else:
                        flux_z_prev = 0
                    
                    # Update saturation
                    div_flux = ((flux_x - flux_x_prev) / self.reservoir.dx +
                               (flux_y - flux_y_prev) / self.reservoir.dy +
                               (flux_z - flux_z_prev) / self.reservoir.dz)
                    
                    oil_sat_new[i, j, k] -= dt * div_flux / self.reservoir.porosity[i, j, k]
        
        # Apply similar update for water saturation (simplified)
        water_sat_new = self.reservoir.water_saturation.copy()
        water_sat_new = np.maximum(water_sat_new - 0.001 * dt, 0.01)  # Simplified update
        
        # Update saturations
        self.reservoir.oil_saturation = np.clip(oil_sat_new, 0.01, 0.99)
        self.reservoir.water_saturation = np.clip(water_sat_new, 0.01, 0.99)
        self.reservoir.gas_saturation = 1.0 - self.reservoir.oil_saturation - self.reservoir.water_saturation
        self.reservoir.gas_saturation = np.maximum(self.reservoir.gas_saturation, 0.0)
    
    def _calculate_fractional_flow(self, phase):
        """Calculate fractional flow for given phase"""
        fractional_flow = np.zeros((self.nx, self.ny, self.nz))
        
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if phase == 'oil':
                        sat = self.reservoir.oil_saturation[i, j, k]
                    elif phase == 'gas':
                        sat = self.reservoir.gas_saturation[i, j, k]
                    elif phase == 'water':
                        sat = self.reservoir.water_saturation[i, j, k]
                    else:
                        continue
                    
                    kr = self.reservoir.get_relative_permeability(sat, phase)
                    mu = getattr(self.reservoir.pvt, f'{phase}_viscosity')(self.reservoir.pressure[i, j, k])
                    
                    mobility = kr / mu
                    
                    # Total mobility (simplified)
                    total_mobility = 1e-12  # Placeholder
                    
                    if total_mobility > 0:
                        fractional_flow[i, j, k] = mobility / total_mobility
        
        return fractional_flow
