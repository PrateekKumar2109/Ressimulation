import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from reservoir_simulator import ReservoirSimulator
from visualization import ReservoirVisualizer

def create_interactive_grid(nx, ny, nz, selected_wells=None):
    """
    Create an interactive grid for well placement
    """
    if selected_wells is None:
        selected_wells = []
    
    # Create a 2D grid for the top layer
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    # Create base grid
    fig = go.Figure()
    
    # Add grid cells as scatter plot
    fig.add_trace(go.Scatter(
        x=X.flatten(),
        y=Y.flatten(),
        mode='markers',
        marker=dict(
            size=25,
            color='lightblue',
            symbol='square',
            line=dict(width=2, color='darkblue')
        ),
        name='Available Cells',
        hovertemplate='Grid Cell: (%{x}, %{y})<br>Click to select<extra></extra>',
        showlegend=True
    ))
    
    # Add existing wells
    for well in selected_wells:
        color = 'red' if well['type'] == 'producer' else 'blue'
        symbol = 'circle' if well['type'] == 'producer' else 'diamond'
        
        fig.add_trace(go.Scatter(
            x=[well['i']],
            y=[well['j']],
            mode='markers+text',
            marker=dict(
                size=20,
                color=color,
                symbol=symbol,
                line=dict(width=3, color='black')
            ),
            text=[well['name']],
            textposition='top center',
            name=f"{well['name']}",
            hovertemplate=f"Well: {well['name']}<br>Type: {well['type']}<br>Location: ({well['i']}, {well['j']}, {well['k']})<extra></extra>",
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Reservoir Grid ({nx} × {ny} × {nz}) - Interactive Well Placement",
        xaxis_title="Grid X Index",
        yaxis_title="Grid Y Index",
        width=700,
        height=600,
        showlegend=True,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, nx-0.5],
            showgrid=True,
            gridwidth=1,
            gridcolor='gray'
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, ny-0.5],
            scaleanchor="x",
            scaleratio=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='gray'
        ),
        plot_bgcolor='white'
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="3D Black Oil Reservoir Simulator",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛢️ 3D Black Oil Reservoir Simulation")
    st.markdown("### Advanced Multiphase Flow Dynamics Simulator")
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'selected_wells' not in st.session_state:
        st.session_state.selected_wells = []
    if 'grid_initialized' not in st.session_state:
        st.session_state.grid_initialized = False
    if 'selected_cell' not in st.session_state:
        st.session_state.selected_cell = None
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("Simulation Parameters")
        
        # Grid dimensions
        st.subheader("Grid Dimensions")
        col1, col2, col3 = st.columns(3)
        with col1:
            nx = st.number_input("Grid X", min_value=5, max_value=50, value=10, step=1, key="nx")
        with col2:
            ny = st.number_input("Grid Y", min_value=5, max_value=50, value=10, step=1, key="ny")
        with col3:
            nz = st.number_input("Grid Z", min_value=3, max_value=20, value=5, step=1, key="nz")
        
        # Physical dimensions
        st.subheader("Physical Dimensions")
        length_x = st.number_input("Length X (m)", value=100.0)
        length_y = st.number_input("Length Y (m)", value=100.0)
        length_z = st.number_input("Length Z (m)", value=10.0)
        
        # Rock properties
        st.subheader("Rock Properties")
        porosity = st.number_input("Porosity", min_value=0.1, max_value=0.4, value=0.2, step=0.01, format="%.3f")
        permeability_x = st.number_input("Permeability X (mD)", value=100.0)
        permeability_y = st.number_input("Permeability Y (mD)", value=100.0)
        permeability_z = st.number_input("Permeability Z (mD)", value=10.0)
        
        # Fluid properties
        st.subheader("Fluid Properties")
        initial_pressure = st.number_input("Initial Pressure (psia)", value=3000.0)
        oil_saturation = st.slider("Initial Oil Saturation", 0.1, 0.9, 0.7)
        water_saturation = st.slider("Initial Water Saturation", 0.1, 0.5, 0.3)
        
        # Simulation parameters
        st.subheader("Simulation Control")
        total_time = st.number_input("Total Simulation Time (days)", value=1000.0)
        initial_dt = st.number_input("Initial Time Step (days)", value=1.0)
        max_dt = st.number_input("Maximum Time Step (days)", value=30.0)
        
        # Button to create grid
        if st.button("🏗️ Create Grid", type="secondary"):
            st.session_state.grid_initialized = True
            st.session_state.selected_wells = []  # Reset wells when grid changes
            st.session_state.selected_cell = None
            st.success("Grid created! Now you can place wells.")
            st.rerun()
        
        # Well management section (only show if grid is initialized)
        if st.session_state.grid_initialized:
            st.subheader("🎯 Well Placement")
            
            # Manual coordinate input as backup
            with st.expander("Manual Well Placement", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    manual_x = st.number_input("X Coordinate", min_value=0, max_value=int(nx)-1, value=0, key="manual_x")
                    manual_y = st.number_input("Y Coordinate", min_value=0, max_value=int(ny)-1, value=0, key="manual_y")
                with col2:
                    manual_z = st.number_input("Z Layer", min_value=0, max_value=int(nz)-1, value=0, key="manual_z")
                
                well_name = st.text_input("Well Name", value=f"WELL_{len(st.session_state.selected_wells)+1}")
                well_type = st.selectbox("Well Type", ["producer", "injector"])
                
                if well_type == "producer":
                    well_rate = st.number_input("Production Rate (STB/day)", value=500.0, min_value=0.0) * -1
                else:
                    well_rate = st.number_input("Injection Rate (STB/day)", value=300.0, min_value=0.0)
                
                if st.button("➕ Add Well at Coordinates", type="primary"):
                    # Check if location is already occupied
                    occupied = any(well['i'] == manual_x and well['j'] == manual_y and well['k'] == manual_z 
                                 for well in st.session_state.selected_wells)
                    
                    if not occupied:
                        new_well = {
                            'name': well_name,
                            'type': well_type,
                            'i': int(manual_x),
                            'j': int(manual_y),
                            'k': int(manual_z),
                            'rate': well_rate,
                            'radius': 0.1,
                            'skin': 0.0
                        }
                        st.session_state.selected_wells.append(new_well)
                        st.success(f"Added {well_name} at ({manual_x}, {manual_y}, {manual_z})")
                        st.rerun()
                    else:
                        st.error("Location already occupied by another well!")
            
            # Display current wells
            if st.session_state.selected_wells:
                st.markdown("**📍 Current Wells:**")
                for i, well in enumerate(st.session_state.selected_wells):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        icon = "🔴" if well['type'] == 'producer' else "🔵"
                        st.markdown(f"{icon} **{well['name']}** ({well['type']})")
                        st.markdown(f"    Location: ({well['i']}, {well['j']}, {well['k']})")
                        st.markdown(f"    Rate: {abs(well['rate']):.0f} STB/day")
                    with col2:
                        if st.button("🗑️", key=f"remove_well_{i}", help="Remove well"):
                            st.session_state.selected_wells.pop(i)
                            st.rerun()
                
                st.markdown("---")
        
        # Initialize simulation button (only show if wells are placed)
        if st.session_state.selected_wells:
            if st.button("🚀 Initialize Simulation", type="primary"):
                with st.spinner("Initializing reservoir simulation..."):
                    try:
                        st.session_state.simulator = ReservoirSimulator(
                            nx=int(nx), ny=int(ny), nz=int(nz),
                            length_x=length_x, length_y=length_y, length_z=length_z,
                            porosity=porosity,
                            permeability_x=permeability_x,
                            permeability_y=permeability_y,
                            permeability_z=permeability_z,
                            initial_pressure=initial_pressure,
                            initial_oil_saturation=oil_saturation,
                            initial_water_saturation=water_saturation
                        )
                        
                        # Convert and assign wells directly with proper unit conversion
                        converted_wells = []
                        for well in st.session_state.selected_wells:
                            converted_well = {
                                'name': well['name'],
                                'type': well['type'],
                                'i': well['i'],
                                'j': well['j'],
                                'k': well['k'],
                                'rate': well['rate'] * 0.159 if well['type'] == 'injector' else well['rate'] * 0.159  # Convert STB/day to m³/day
                            }
                            converted_wells.append(converted_well)
                        
                        # Directly assign wells to simulator.wells
                        st.session_state.simulator.wells = converted_wells
                        
                        st.success(f"Simulation initialized with {len(converted_wells)} wells!")
                    except Exception as e:
                        st.error(f"Error initializing simulation: {str(e)}")
    
    # Main content area
    if not st.session_state.grid_initialized:
        st.info("👆 Please configure grid parameters in the sidebar and click **'Create Grid'** to begin.")
        
        # Display equations and theory
        st.subheader("Black Oil Reservoir Simulation Theory")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Governing Equations")
            st.latex(r'''
            \frac{\partial}{\partial t}\left(\frac{\phi S_o B_o}{B_{oi}}\right) + \nabla \cdot \left(\frac{\rho_o \vec{v}_o}{B_o}\right) = q_o
            ''')
            st.markdown("**Oil Conservation Equation**")
            
            st.latex(r'''
            \frac{\partial}{\partial t}\left(\frac{\phi S_g B_g}{B_{gi}}\right) + \nabla \cdot \left(\frac{\rho_g \vec{v}_g}{B_g}\right) = q_g
            ''')
            st.markdown("**Gas Conservation Equation**")
            
            st.latex(r'''
            \frac{\partial}{\partial t}\left(\frac{\phi S_w B_w}{B_{wi}}\right) + \nabla \cdot \left(\frac{\rho_w \vec{v}_w}{B_w}\right) = q_w
            ''')
            st.markdown("**Water Conservation Equation**")
        
        with col2:
            st.markdown("#### Darcy's Law")
            st.latex(r'''
            \vec{v}_\alpha = -\frac{k k_{r\alpha}}{\mu_\alpha}\left(\nabla P_\alpha - \rho_\alpha g \nabla D\right)
            ''')
            st.markdown("**Phase Velocity (α = oil, gas, water)**")
            
            st.markdown("#### Constraints")
            st.latex(r'''S_o + S_g + S_w = 1''')
            st.markdown("**Saturation Constraint**")
            
            st.latex(r'''P_g = P_o + P_{cog}''')
            st.latex(r'''P_o = P_w + P_{cow}''')
            st.markdown("**Capillary Pressure Relations**")
    
    elif st.session_state.simulator is None:
        # Show grid for well placement
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Reservoir Grid Layout")
            st.markdown("**Instructions:** Use the manual well placement controls in the sidebar to add wells to specific grid coordinates.")
            
            # Create and display interactive grid
            grid_fig = create_interactive_grid(int(nx), int(ny), int(nz), st.session_state.selected_wells)
            st.plotly_chart(grid_fig, use_container_width=True, key="reservoir_grid")
        
        with col2:
            st.markdown("#### 📊 Grid Information")
            st.info(f"""
            **Dimensions:** {int(nx)} × {int(ny)} × {int(nz)}
            
            **Total Cells:** {int(nx) * int(ny) * int(nz):,}
            
            **Physical Size:** 
            - {length_x} × {length_y} × {length_z} m
            
            **Cell Size:** 
            - {length_x/nx:.1f} × {length_y/ny:.1f} × {length_z/nz:.1f} m
            """)
            
            st.markdown("#### 🎯 Legend")
            st.markdown("""
            - 🟦 **Blue Squares**: Available grid cells
            - 🔴 **Red Circles**: Production wells  
            - 🔵 **Blue Diamonds**: Injection wells
            """)
            
            if st.session_state.selected_wells:
                st.markdown("#### ✅ Next Step")
                st.success("Wells placed! Click **'Initialize Simulation'** in the sidebar to proceed.")
            else:
                st.markdown("#### 🎯 Next Step")
                st.warning("Add wells using the sidebar controls, then initialize the simulation.")
    
    else:
        # Simulation controls and results
        st.subheader("🎮 Simulation Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ Run Simulation", disabled=st.session_state.is_running):
                st.session_state.is_running = True
                st.rerun()
        
        with col2:
            if st.button("⏸️ Pause Simulation"):
                st.session_state.is_running = False
        
        with col3:
            if st.button("🔄 Reset Simulation"):
                st.session_state.simulator.reset()
                st.session_state.simulation_results = None
        
        with col4:
            time_step = st.number_input("Current Time Step (days)", 
                                      value=st.session_state.simulator.current_time, 
                                      disabled=True)
        
        # Run simulation if requested
        if st.session_state.is_running:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run simulation steps
            max_steps = int(total_time / initial_dt)
            
            for step in range(min(10, max_steps)):  # Run 10 steps at a time for responsiveness
                if not st.session_state.is_running:
                    break
                
                try:
                    st.session_state.simulator.step(initial_dt, max_dt)
                    progress = st.session_state.simulator.current_time / total_time
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Time: {st.session_state.simulator.current_time:.1f} days")
                    
                    if st.session_state.simulator.current_time >= total_time:
                        st.session_state.is_running = False
                        st.success("Simulation completed!")
                        break
                        
                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
                    st.session_state.is_running = False
                    break
            
            if st.session_state.is_running:
                time.sleep(0.1)  # Small delay for UI responsiveness
                st.rerun()
        
        # Display simulation results
        if st.session_state.simulator.current_time > 0:
            st.subheader("📈 Simulation Results")
            
            # Create visualizer
            visualizer = ReservoirVisualizer(st.session_state.simulator)
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Pressure Field", "Saturation Distribution", "Production Data", "Material Balance"])
            
            with tab1:
                st.markdown("#### Pressure Distribution")
                layer_select = st.selectbox("Select Layer", range(int(nz)), key="pressure_layer")
                fig_pressure = visualizer.plot_pressure_field(layer=layer_select)
                st.plotly_chart(fig_pressure, use_container_width=True)
            
            with tab2:
                st.markdown("#### Saturation Distribution")
                phase_select = st.selectbox("Select Phase", ["Oil", "Gas", "Water"], key="saturation_phase")
                layer_select_sat = st.selectbox("Select Layer", range(int(nz)), key="saturation_layer")
                fig_saturation = visualizer.plot_saturation_field(phase_select.lower(), layer=layer_select_sat)
                st.plotly_chart(fig_saturation, use_container_width=True)
            
            with tab3:
                st.markdown("#### Production History")
                if len(st.session_state.simulator.production_history) > 0:
                    fig_production = visualizer.plot_production_history()
                    st.plotly_chart(fig_production, use_container_width=True)
                else:
                    st.info("No production data available yet.")
            
            with tab4:
                st.markdown("#### Material Balance")
                if len(st.session_state.simulator.material_balance_history) > 0:
                    fig_material = visualizer.plot_material_balance()
                    st.plotly_chart(fig_material, use_container_width=True)
                    
                    # Display current material balance
                    current_balance = st.session_state.simulator.material_balance_history[-1]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Oil in Place", f"{current_balance['oil_in_place']:.0f}", "STB")
                    with col2:
                        st.metric("Gas in Place", f"{current_balance['gas_in_place']:.0f}", "SCF")
                    with col3:
                        st.metric("Water in Place", f"{current_balance['water_in_place']:.0f}", "STB")
                else:
                    st.info("No material balance data available yet.")
        
        # Display grid and well information
        with st.expander("📋 Grid and Well Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Grid Information")
                st.write(f"Grid dimensions: {int(nx)} × {int(ny)} × {int(nz)}")
                st.write(f"Total cells: {int(nx) * int(ny) * int(nz):,}")
                st.write(f"Physical dimensions: {length_x} × {length_y} × {length_z} m")
                st.write(f"Cell size: {length_x/nx:.1f} × {length_y/ny:.1f} × {length_z/nz:.1f} m")
            
            with col2:
                st.markdown("#### Well Information")
                if hasattr(st.session_state.simulator, 'wells') and st.session_state.simulator.wells:
                    for well in st.session_state.simulator.wells:
                        st.write(f"**{well['name']}** ({well['type']})")
                        st.write(f"Location: ({well['i']}, {well['j']}, {well['k']})")
                        st.write(f"Rate: {abs(well['rate']):.1f} m³/day")
                        st.write("---")
                else:
                    st.write("No wells configured")

if __name__ == "__main__":
    main()
