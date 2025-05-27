import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class ReservoirVisualizer:
    """
    Visualization class for reservoir simulation results
    
    Provides 2D and 3D plotting capabilities for pressure, saturation,
    and production data visualization.
    """
    
    def __init__(self, reservoir):
        self.reservoir = reservoir
        self.x, self.y, self.z = reservoir.get_grid_coordinates()
    
    def plot_pressure_field(self, layer=None, time_step=None):
        """
        Plot pressure field as 2D heatmap for a specific layer
        
        Args:
            layer: Layer index (0 to nz-1). If None, plot middle layer
            time_step: Time step (for future time series plotting)
        
        Returns:
            Plotly figure object
        """
        if layer is None:
            layer = self.reservoir.nz // 2
        
        layer = max(0, min(layer, self.reservoir.nz - 1))
        
        # Extract pressure data for the layer
        pressure_layer = self.reservoir.pressure[:, :, layer] / 6895  # Convert to psia
        
        # Create mesh coordinates
        x_mesh, y_mesh = np.meshgrid(self.x, self.y, indexing='ij')
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            x=self.y,
            y=self.x,
            z=pressure_layer,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(
                    text="Pressure (psia)",
                    side="right"  # Replaces titleside
                )
            ),
            hovertemplate="X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Pressure: %{z:.1f} psia<extra></extra>"
        ))
        # Add well locations
        for well in self.reservoir.wells:
            if well['k'] == layer:
                x_well = self.y[well['j']]
                y_well = self.x[well['i']]
                
                symbol = 'circle' if well['type'] == 'producer' else 'square'
                color = 'red' if well['type'] == 'producer' else 'blue'
                
                fig.add_trace(go.Scatter(
                    x=[x_well],
                    y=[y_well],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=12,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    name=well['name'],
                    hovertemplate=f"{well['name']}<br>Type: {well['type']}<extra></extra>"
                ))
        
        fig.update_layout(
            title=f"Pressure Field - Layer {layer+1} (Time: {self.reservoir.current_time:.1f} days)",
            xaxis_title="Y Distance (m)",
            yaxis_title="X Distance (m)",
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_saturation_field(self, phase='oil', layer=None):
        """
        Plot saturation field for a specific phase and layer
        
        Args:
            phase: 'oil', 'gas', or 'water'
            layer: Layer index. If None, plot middle layer
        
        Returns:
            Plotly figure object
        """
        if layer is None:
            layer = self.reservoir.nz // 2
        
        layer = max(0, min(layer, self.reservoir.nz - 1))
        
        # Get saturation data
        if phase.lower() == 'oil':
            saturation_data = self.reservoir.oil_saturation[:, :, layer]
            colorscale = 'Greens'
        elif phase.lower() == 'gas':
            saturation_data = self.reservoir.gas_saturation[:, :, layer]
            colorscale = 'Reds'
        elif phase.lower() == 'water':
            saturation_data = self.reservoir.water_saturation[:, :, layer]
            colorscale = 'Blues'
        else:
            raise ValueError("Phase must be 'oil', 'gas', or 'water'")
        
        fig = go.Figure()
        
        # Add saturation heatmap
        fig.add_trace(go.Heatmap(
            x=self.y,
            y=self.x,
            z=saturation_data,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(
                    text=f"{phase.capitalize()} Saturation",
                    side="right"
                )
            ),
            hovertemplate="X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Saturation: %{z:.3f}<extra></extra>"
        ))        
        # Add well locations
        for well in self.reservoir.wells:
            if well['k'] == layer:
                x_well = self.y[well['j']]
                y_well = self.x[well['i']]
                
                symbol = 'circle' if well['type'] == 'producer' else 'square'
                color = 'red' if well['type'] == 'producer' else 'blue'
                
                fig.add_trace(go.Scatter(
                    x=[x_well],
                    y=[y_well],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=12,
                        color=color,
                        line=dict(width=2, color='white')
                    ),
                    name=well['name'],
                    showlegend=False,
                    hovertemplate=f"{well['name']}<br>Type: {well['type']}<extra></extra>"
                ))
        
        fig.update_layout(
            title=f"{phase.capitalize()} Saturation - Layer {layer+1} (Time: {self.reservoir.current_time:.1f} days)",
            xaxis_title="Y Distance (m)",
            yaxis_title="X Distance (m)",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_3d_pressure(self, opacity=0.6):
        """
        Plot 3D pressure field as volume rendering
        
        Args:
            opacity: Opacity for volume rendering
        
        Returns:
            Plotly figure object
        """
        # Create 3D mesh
        x_3d, y_3d, z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Flatten arrays for plotting
        x_flat = x_3d.flatten()
        y_flat = y_3d.flatten()
        z_flat = z_3d.flatten()
        pressure_flat = (self.reservoir.pressure / 6895).flatten()  # Convert to psia
        
        fig = go.Figure()
        
        # Add 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            mode='markers',
            marker=dict(
                size=2,
                color=pressure_flat,
                colorscale='Viridis',
                opacity=opacity,
                colorbar=dict(title="Pressure (psia)")
            ),
            hovertemplate="X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Z: %{z:.0f} m<br>Pressure: %{marker.color:.1f} psia<extra></extra>"
        ))
        
        # Add wells
        for well in self.reservoir.wells:
            x_well = self.x[well['i']]
            y_well = self.y[well['j']]
            z_well = self.z[well['k']]
            
            color = 'red' if well['type'] == 'producer' else 'blue'
            
            fig.add_trace(go.Scatter3d(
                x=[x_well],
                y=[y_well],
                z=[z_well],
                mode='markers',
                marker=dict(
                    size=8,
                    color=color,
                    symbol='circle'
                ),
                name=well['name']
            ))
        
        fig.update_layout(
            title=f"3D Pressure Field (Time: {self.reservoir.current_time:.1f} days)",
            scene=dict(
                xaxis_title="X Distance (m)",
                yaxis_title="Y Distance (m)",
                zaxis_title="Z Distance (m)"
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_production_history(self):
        """
        Plot production history for all phases
        
        Returns:
            Plotly figure object
        """
        if not self.reservoir.production_history:
            # Create empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No production data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Production History")
            return fig
        
        # Convert to DataFrame
        df = pd.DataFrame(self.reservoir.production_history)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Oil Production", "Gas Production", "Water Production"),
            vertical_spacing=0.08
        )
        
        # Oil production
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['oil_rate'],
                mode='lines',
                name='Oil Rate',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Gas production
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['gas_rate'],
                mode='lines',
                name='Gas Rate',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Water production
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['water_rate'],
                mode='lines',
                name='Water Rate',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (days)", row=3, col=1)
        fig.update_yaxes(title_text="Oil Rate (STB/day)", row=1, col=1)
        fig.update_yaxes(title_text="Gas Rate (SCF/day)", row=2, col=1)
        fig.update_yaxes(title_text="Water Rate (STB/day)", row=3, col=1)
        
        fig.update_layout(
            title="Production History",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_material_balance(self):
        """
        Plot material balance history
        
        Returns:
            Plotly figure object
        """
        if not self.reservoir.material_balance_history:
            fig = go.Figure()
            fig.add_annotation(
                text="No material balance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title="Material Balance")
            return fig
        
        # Convert to DataFrame
        df = pd.DataFrame(self.reservoir.material_balance_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Oil in Place", "Gas in Place", "Water in Place", "Average Pressure"),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Oil in place
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['oil_in_place'],
                mode='lines',
                name='Oil in Place',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Gas in place
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['gas_in_place'],
                mode='lines',
                name='Gas in Place',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Water in place
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['water_in_place'],
                mode='lines',
                name='Water in Place',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Average pressure
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['average_pressure'],
                mode='lines',
                name='Average Pressure',
                line=dict(color='purple', width=2)
            ),
            row=2, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=2)
        fig.update_yaxes(title_text="Oil (STB)", row=1, col=1)
        fig.update_yaxes(title_text="Gas (SCF)", row=1, col=2)
        fig.update_yaxes(title_text="Water (STB)", row=2, col=1)
        fig.update_yaxes(title_text="Pressure (psia)", row=2, col=2)
        
        fig.update_layout(
            title="Material Balance History",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_well_performance(self, well_name):
        """
        Plot individual well performance
        
        Args:
            well_name: Name of the well
        
        Returns:
            Plotly figure object
        """
        # This would require storing well-specific production data
        # For now, return a placeholder
        fig = go.Figure()
        fig.add_annotation(
            text=f"Well performance plot for {well_name}\n(Feature to be implemented)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=f"Well Performance - {well_name}")
        return fig
