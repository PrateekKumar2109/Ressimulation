import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import cohere
import os

# Initialize Cohere client with proper error handling
def initialize_cohere():
    """Initialize Cohere client with proper error handling"""
    try:
        # Try to get API key from Streamlit secrets first, then environment variables
        cohere_api_key = "vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg"
        
        if not cohere_api_key:
            st.error("❌ Cohere API key not found. Please set COHERE_API_KEY in Streamlit secrets or environment variables.")
            return None
        
        co = cohere.Client(cohere_api_key)
        
        # Test the connection
        co.generate(
            model="command",
            prompt="Test connection",
            max_tokens=5
        )
        
        st.success("✅ Cohere API connected successfully!")
        return co
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Cohere: {str(e)}")
        return None

# Context about the web app for the assistant
APP_CONTEXT = """
This is a 3D Black Oil Reservoir Simulation web app built with Streamlit. It simulates multiphase flow dynamics in a reservoir, allowing users to configure parameters such as grid dimensions, physical dimensions, rock properties (porosity, permeability), fluid properties (initial pressure, oil and water saturation), and simulation controls (total time, time steps). Users can also set up wells (producers and injectors) with specific rates. The app displays governing equations, Darcy's Law, and constraints for black oil simulation. After initializing and running the simulation, it visualizes results like pressure fields, saturation distributions, production history, and material balance using Plotly charts. The app includes interactive controls to run, pause, or reset the simulation.
"""

def get_assistant_response(co, user_input):
    """Generate a response using Cohere's command model, including app context."""
    if co is None:
        return "Cohere client not initialized. Please check your API key."
    
    prompt = f"{APP_CONTEXT}\n\nUser question: {user_input}\n\nAnswer:"
    try:
        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            k=0,
            stop_sequences=[],
            return_likelihoods="NONE"
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

def summarize_text(co, text):
    """Summarize text using Cohere's summarize model."""
    if co is None:
        return "Cohere client not initialized. Please check your API key."
    
    try:
        response = co.summarize(
            text=text,
            length="medium",
            format="paragraph",
            extractiveness="medium"
        )
        return response.summary
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

# Import your original reservoir simulator and visualizer modules
try:
    from reservoir_simulator import ReservoirSimulator
    from visualization import ReservoirVisualizer
    MODULES_AVAILABLE = True
except ImportError:
    st.error("❌ reservoir_simulator.py and/or visualization.py modules not found. Please ensure these files are in your project directory.")
    MODULES_AVAILABLE = False

def main():
    st.set_page_config(
        page_title="3D Black Oil Reservoir Simulator",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛢️ 3D Black Oil Reservoir Simulation")
    st.markdown("### Advanced Multiphase Flow Dynamics Simulator")
    
    # Initialize Cohere client
    if 'cohere_client' not in st.session_state:
        st.session_state.cohere_client = initialize_cohere()
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'assistant_history' not in st.session_state:
        st.session_state.assistant_history = []
    
    # Tabs for Simulation and Assistant
    tab1, tab2 = st.tabs(["🔧 Simulation", "🧠 AI Assistant"])
    
    with tab1:
        # Sidebar for parameters
        with st.sidebar:
            st.header("Simulation Parameters")
            
            # Grid dimensions
            st.subheader("Grid Dimensions")
            col1, col2, col3 = st.columns(3)
            with col1:
                nx = st.number_input("Grid X", min_value=5, max_value=50, value=20, step=1)
            with col2:
                ny = st.number_input("Grid Y", min_value=5, max_value=50, value=20, step=1)
            with col3:
                nz = st.number_input("Grid Z", min_value=3, max_value=20, value=5, step=1)
            
            # Physical dimensions
            st.subheader("Physical Dimensions")
            length_x = st.number_input("Length X (m)", value=1000.0)
            length_y = st.number_input("Length Y (m)", value=1000.0)
            length_z = st.number_input("Length Z (m)", value=50.0)
            
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
            
            # Well parameters
            st.subheader("Well Configuration")
            num_producers = st.slider("Number of Producers", 1, 5, 2)
            num_injectors = st.slider("Number of Injectors", 0, 3, 1)
            production_rate = st.number_input("Production Rate (STB/day)", value=500.0)
            injection_rate = st.number_input("Injection Rate (STB/day)", value=300.0)
            
            # Initialize simulation button
            if st.button("Initialize Simulation", type="primary"):
                if not MODULES_AVAILABLE:
                    st.error("Cannot initialize simulation. Required modules are missing.")
                else:
                    with st.spinner("Initializing reservoir simulation..."):
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
                        
                        # Add wells
                        st.session_state.simulator.add_wells(
                            num_producers=num_producers,
                            num_injectors=num_injectors,
                            production_rate=production_rate,
                            injection_rate=injection_rate
                        )
                        
                        st.success("Simulation initialized successfully!")
        
        # Main content area
        if st.session_state.simulator is None:
            st.info("Please configure parameters in the sidebar and click 'Initialize Simulation' to begin.")
            
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
        
        else:
            # Simulation controls
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
                st.subheader("Simulation Results")
                
                # Create visualizer
                visualizer = ReservoirVisualizer(st.session_state.simulator)
                
                # Tabs for different visualizations
                tab_results1, tab_results2, tab_results3, tab_results4 = st.tabs(["Pressure Field", "Saturation Distribution", "Production Data", "Material Balance"])
                
                with tab_results1:
                    st.markdown("#### Pressure Distribution")
                    layer_select = st.selectbox("Select Layer", range(int(nz)), key="pressure_layer")
                    fig_pressure = visualizer.plot_pressure_field(layer=layer_select)
                    st.plotly_chart(fig_pressure, use_container_width=True)
                
                with tab_results2:
                    st.markdown("#### Saturation Distribution")
                    phase_select = st.selectbox("Select Phase", ["Oil", "Gas", "Water"], key="saturation_phase")
                    layer_select_sat = st.selectbox("Select Layer", range(int(nz)), key="saturation_layer")
                    fig_saturation = visualizer.plot_saturation_field(phase_select.lower(), layer=layer_select_sat)
                    st.plotly_chart(fig_saturation, use_container_width=True)
                
                with tab_results3:
                    st.markdown("#### Production History")
                    if len(st.session_state.simulator.production_history) > 0:
                        fig_production = visualizer.plot_production_history()
                        st.plotly_chart(fig_production, use_container_width=True)
                    else:
                        st.info("No production data available yet.")
                
                with tab_results4:
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
            with st.expander("Grid and Well Information"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Grid Information")
                    st.write(f"Grid dimensions: {int(nx)} × {int(ny)} × {int(nz)}")
                    st.write(f"Total cells: {int(nx) * int(ny) * int(nz):,}")
                    st.write(f"Physical dimensions: {length_x} × {length_y} × {length_z} m")
                    st.write(f"Cell size: {length_x/nx:.1f} × {length_y/ny:.1f} × {length_z/nz:.1f} m")
                
                with col2:
                    st.markdown("#### Well Information")
                    for well in st.session_state.simulator.wells:
                        st.write(f"**{well['name']}** ({well['type']})")
                        st.write(f"Location: ({well['i']}, {well['j']}, {well['k']})")
                        st.write(f"Rate: {well['rate']:.1f} STB/day")
                        st.write("---")
    
    with tab2:
        st.header("🧠 AI Assistant")
        st.markdown("Ask questions about reservoir simulation, petroleum engineering, or this app's functionality.")
        
        # Display Cohere connection status
        if st.session_state.cohere_client:
            st.success("✅ Cohere AI is connected and ready!")
        else:
            st.error("❌ Cohere AI is not connected. Please check your API key.")
            st.info("💡 **How to fix:** Add your Cohere API key to Streamlit secrets or environment variables with key 'COHERE_API_KEY'")
        
        # Chat interface
        with st.container():
            user_input = st.text_area("Your question:", 
                                    placeholder="e.g., What is the difference between oil and water saturation?",
                                    height=100)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                ask_button = st.button("🤖 Ask Assistant", type="primary", disabled=not st.session_state.cohere_client)
            
            with col2:
                summarize_option = st.checkbox("📝 Summarize response", 
                                             help="Use Cohere's summarize model to condense the response")
        
        if ask_button and user_input:
            with st.spinner("🤔 Thinking..."):
                # Store user input in history
                st.session_state.assistant_history.append({"role": "user", "message": user_input})
                
                # Get response from Cohere
                if summarize_option:
                    response = summarize_text(st.session_state.cohere_client, user_input)
                else:
                    response = get_assistant_response(st.session_state.cohere_client, user_input)
                
                # Store assistant response in history
                st.session_state.assistant_history.append({"role": "assistant", "message": response})
        
        # Display chat history
        if st.session_state.assistant_history:
            st.subheader("💬 Conversation History")
            
            # Reverse order to show latest first
            for i, chat in enumerate(reversed(st.session_state.assistant_history)):
                if chat["role"] == "user":
                    with st.chat_message("user"):
                        st.write(chat['message'])
                else:
                    with st.chat_message("assistant"):
                        st.write(chat['message'])
            
            # Clear history button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.assistant_history = []
                st.rerun()
        
        # Quick questions section
        st.subheader("💡 Quick Questions")
        st.markdown("Try asking about these topics:")
        
        quick_questions = [
            "What is reservoir simulation?",
            "Explain Darcy's law in simple terms",
            "What's the difference between oil and gas saturation?",
            "How do production wells work?",
            "What is porosity and permeability?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(f"❓ {question}", key=f"quick_{i}"):
                    if st.session_state.cohere_client:
                        with st.spinner("Getting answer..."):
                            st.session_state.assistant_history.append({"role": "user", "message": question})
                            response = get_assistant_response(st.session_state.cohere_client, question)
                            st.session_state.assistant_history.append({"role": "assistant", "message": response})
                            st.rerun()

if __name__ == "__main__":
    main()
