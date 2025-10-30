import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import matplotlib.pyplot as plt
import random
import io
import google.generativeai as genai

#Configurating Google Gemini
try:
    genai.configure(api_key= st.secrets['GOOGLE_API_KEY'])
    model = genai.GenerativeModel("gemini-2.5-pro")
    llm_available = True
except Exception as e:
    st.warning("LLM could not be configured right now, please try later")
    llm_available = False

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Supply Chain Optimization")

# --- Helper Function for Excel Download ---
@st.cache_data # Cache the generated Excel data
def to_excel(df_dict):
    """Converts a dictionary of dataframes to an Excel file in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=True)
    processed_data = output.getvalue()
    return processed_data

# --- Function to Load/Generate Dummy Data ---
@st.cache_data # Cache the dummy dataframes
def load_dummy_data():
    """Generates plausible dummy data based on the template structure."""
    locations = ['USA', 'GERMANY', 'JAPAN', 'BRAZIL', 'INDIA']
    capacities = ['LOW', 'HIGH']

    # More realistic dummy data
    df_freight = pd.DataFrame({
        'USA': [0, 1750, 1100, 2300, 1254],
        'GERMANY': [1905, 0, 1231, 2892, 1439],
        'JAPAN': [2200, 3250, 0, 6230, 2050],
        'BRAZIL': [2350, 3150, 4000, 0, 4250],
        'INDIA': [1950, 2200, 3500, 4200, 0]
    }, index=locations)

    df_variable = pd.DataFrame({ # Manufacturing Variable Cost
        'USA': [12]*5, 'GERMANY': [13]*5, 'JAPAN': [10]*5, 'BRAZIL': [8]*5, 'INDIA': [5]*5
    }, index=locations)
    # Replicate columns to match structure (though function only uses index)
    df_variable = df_variable.reindex(columns=locations, method='ffill')


    df_fixed = pd.DataFrame({
        'LOW': [6500, 4980, 6230, 3230, 2110],
        'HIGH': [9500, 7270, 9100, 4730, 6160]
    }, index=locations)

    df_capacity = pd.DataFrame({
        'LOW': [500]*5, 'HIGH': [1500]*5
    }, index=locations)

    df_demand = pd.DataFrame({
        'Demand': [2800000, 90000, 1700000, 145000, 160000]
    }, index=locations)

    return {
        "freight": df_freight,
        "variable": df_variable,
        "fixed": df_fixed,
        "capacity": df_capacity,
        "demand": df_demand
    }


# --- Optimization Function (Copied from Notebook - slightly modified for robustness) ---
def optimization_simulation(fixed_cost_df, variable_cost_df, demand_df, demand_col, capacity_df):
    """Runs the capacitated plant location optimization."""
    try:
        locations = list(demand_df.index)
        plant_capacity_types = list(capacity_df.columns) # Use columns from capacity_df
        plant_name  = [(i , c) for i in locations for c in plant_capacity_types] # Swapped loop order
        production_name = [(i , j) for i in locations for j in locations]

        # Model initialization
        model = LpProblem('Capacitated_Plant_Location_Model', LpMinimize)

        # Creating Decision Variables: Ensure Continuous is lowercase
        x = LpVariable.dicts('production_' , production_name , lowBound=0 , upBound=None , cat=LpContinuous) # Use LpContinuous constant
        y = LpVariable.dicts('plant_' , plant_name , cat=LpBinary) # Use LpBinary constant

        # Objective function
        model += (lpSum([fixed_cost_df.loc[i , c]*y[(i,c)]*1000 for i in locations for c in plant_capacity_types]) +
                  lpSum([variable_cost_df.loc[i,j]*x[(i,j)] for i in locations for j in locations]))

        # Constraints:
        # Demand constraint: Ensure accessing demand correctly
        if isinstance(demand_col, str): # Single column name for initial run
             for j in locations:
                 model += lpSum([x[(i,j)] for i in locations]) == demand_df.loc[j, demand_col]
        else: # Column index for simulation runs (assuming demand_var format)
            for j in locations:
                 model += lpSum([x[(i,j)] for i in locations]) == demand_df.loc[j, demand_col] # access by index/name

        # Capacity constraint
        for i in locations:
            model += lpSum([x[(i,j)] for j in locations]) <= lpSum([capacity_df.loc[i,c]*y[(i,c)]*1000 for c in plant_capacity_types])

        # Solving
        model.solve(PULP_CBC_CMD(msg=0)) # Suppress solver messages in Streamlit

        # Output
        status_out = LpStatus[model.status]
        objective_out = pulp.value(model.objective) if model.status == LpStatusOptimal else None
        y_results = {plant: y[plant].varValue for plant in plant_name} if model.status == LpStatusOptimal else {}
        x_results = {prod: x[prod].varValue for prod in production_name} if model.status == LpStatusOptimal else {}
        fixed = pulp.value(lpSum([fixed_cost_df.loc[i , c]*y[(i,c)]*1000 for i in locations for c in plant_capacity_types])) if model.status == LpStatusOptimal else None
        variable = pulp.value(lpSum([variable_cost_df.loc[i,j]*x[(i,j)] for i in locations for j in locations])) if model.status == LpStatusOptimal else None

        return status_out, objective_out, y_results, x_results, fixed, variable

    except Exception as e:
        st.error(f"Error during optimization: {e}")
        # Print details for debugging in console if needed
        # print(f"Locations: {locations}")
        # print(f"Plant Capacity Types: {plant_capacity_types}")
        # print(f"Fixed Cost Index/Cols: {fixed_cost_df.index}, {fixed_cost_df.columns}")
        # print(f"Var Cost Index/Cols: {variable_cost_df.index}, {variable_cost_df.columns}")
        # print(f"Demand Index/Cols: {demand_df.index}, {demand_df.columns}")
        # print(f"Capacity Index/Cols: {capacity_df.index}, {capacity_df.columns}")
        # print(f"Accessing demand with column: {demand_col}")
        return "Error", None, {}, {}, None, None


# --- Streamlit App Layout ---
st.title("🏭 Supply Chain Network Optimization")
st.write("""
This app determines the optimal plant locations and production flows to minimize total costs (fixed + variable)
based on freight, manufacturing, capacity, and demand data. It also simulates demand uncertainty.
""")

# --- Data Source Selection ---
st.sidebar.header("💾 Data Source")
use_dummy_data = st.sidebar.toggle("Use Dummy Data", value=True) # Default to using dummy data

# --- Template Download Section ---
st.sidebar.header("📥 Download Template Files")
template_instructions = """
**(Only if 'Use Dummy Data' is OFF)**
Download these templates, populate them with your data (keeping the structure the same - **especially index column**),
and upload them below.
"""
st.sidebar.info(template_instructions)

# Generate template data (can be simplified if dummy data is always available)
dummy_data_for_templates = load_dummy_data() # Use the dummy data structure for templates

# *** UPDATED SHEET NAMES BELOW ***
st.sidebar.download_button(
    label="Download Freight Costs Template",
    data=to_excel({'Freight Costs ($ per Container)': dummy_data_for_templates["freight"]}), # Changed sheet name
    file_name="template_freight_costs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.sidebar.download_button(
    label="Download Variable Costs Template",
    data=to_excel({'Variable Costs ($ per Unit)': dummy_data_for_templates["variable"]}), # Changed sheet name
    file_name="template_variable_costs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.sidebar.download_button(
    label="Download Fixed Costs Template",
    data=to_excel({'Fixed Costs (k$ per Month)': dummy_data_for_templates["fixed"]}), # Changed sheet name
    file_name="template_fixed_costs.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.sidebar.download_button(
    label="Download Capacity Template",
    data=to_excel({'Capacity (kUnits per month)': dummy_data_for_templates["capacity"]}), # Changed sheet name
    file_name="template_capacity.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.sidebar.download_button(
    label="Download Demand Template",
    data=to_excel({'Demand (Units per month)': dummy_data_for_templates["demand"]}), # Changed sheet name
    file_name="template_demand.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
# --- File Upload Section (Conditional) ---
uploaded_files = {}
if not use_dummy_data:
    st.sidebar.header("📤 Upload Your Data Files (.xlsx)")
    file_labels = {
        "freight": "Freight Costs ($/Container)",
        "variable": "Variable Costs ($/Unit)",
        "fixed": "Fixed Costs (k$/Month)",
        "capacity": "Capacity (kUnits/month)",
        "demand": "Demand (Units/month)"
    }
    all_files_uploaded = True
    for key, label in file_labels.items():
        uploaded_files[key] = st.sidebar.file_uploader(f"Upload {label}", type=['xlsx'], key=f"upload_{key}")
        if uploaded_files[key] is None:
            all_files_uploaded = False
else:
    all_files_uploaded = False # Not using uploads

# --- Main App Logic ---
data_ready = False
freight_cost_up, manufacturing_variable_cost_up, fixed_cost_up, capacity_up, demand_up = [None]*5

if use_dummy_data:
    st.info("Using pre-loaded dummy data for analysis.")
    dummy_data = load_dummy_data()
    freight_cost_up = dummy_data["freight"]
    manufacturing_variable_cost_up = dummy_data["variable"]
    fixed_cost_up = dummy_data["fixed"]
    capacity_up = dummy_data["capacity"]
    demand_up = dummy_data["demand"]
    data_ready = True

elif all_files_uploaded:
    try:
        # Load data from uploaded files
        freight_cost_up = pd.read_excel(uploaded_files["freight"], index_col=0, engine='openpyxl')
        manufacturing_variable_cost_up = pd.read_excel(uploaded_files["variable"], index_col=0, engine='openpyxl')
        fixed_cost_up = pd.read_excel(uploaded_files["fixed"], index_col=0, engine='openpyxl')
        capacity_up = pd.read_excel(uploaded_files["capacity"], index_col=0, engine='openpyxl')
        demand_up = pd.read_excel(uploaded_files["demand"], index_col=0, engine='openpyxl')
        data_ready = True
        st.success("All files uploaded successfully!")

    except Exception as e:
        st.error(f"Error reading uploaded files: {e}. Please ensure they match the template format.")
        data_ready = False

# --- Proceed only if data is ready ---
if data_ready:
    try:
        # --- Preprocessing ---
        # Basic Validation (Check if indices/columns match roughly)
        locations_demand = list(demand_up.index)
        locations_capacity = list(capacity_up.index)
        locations_fixed = list(fixed_cost_up.index)
        locations_mfg_var = list(manufacturing_variable_cost_up.index)
        locations_freight = list(freight_cost_up.index)

        # Use demand index as the primary source of truth for locations
        # Reindex all dataframes based on demand_up's index
        demand_aligned = demand_up.reindex(index=locations_demand, fill_value=0)
        capacity_aligned = capacity_up.reindex(index=locations_demand, columns=capacity_up.columns, fill_value=0)
        fixed_cost_aligned = fixed_cost_up.reindex(index=locations_demand, columns=capacity_aligned.columns, fill_value=np.inf) # Use capacity cols
        mfg_var_cost_aligned = manufacturing_variable_cost_up.reindex(index=locations_demand, columns=locations_demand, fill_value=np.inf) # Fill with high value if missing
        freight_cost_aligned = freight_cost_up.reindex(index=locations_demand, columns=locations_demand, fill_value=0)

        # Check for potential issues after alignment
        if not (locations_demand == locations_capacity == locations_fixed == locations_mfg_var == locations_freight):
             st.warning("Warning: Row indices (locations) across input files did not perfectly match. Data has been aligned based on the demand file's locations. Missing values might cause issues.")
        if fixed_cost_aligned.isin([np.inf]).any().any() or mfg_var_cost_aligned.isin([np.inf]).any().any():
             st.warning("Warning: Some locations or capacity types seem missing in fixed or variable cost files after alignment (filled with infinity). This might lead to an infeasible solution.")

        # Calculate combined variable cost
        variable_cost_calc = freight_cost_aligned / 1000 + mfg_var_cost_aligned


        # --- Display Input Data ---
        with st.expander("View Processed Input Data"):
            st.write("**Demand (Aligned):**")
            st.dataframe(demand_aligned)
            st.write("**Capacity (Aligned):**")
            st.dataframe(capacity_aligned)
            st.write("**Fixed Costs (Aligned, k$/Month):**")
            st.dataframe(fixed_cost_aligned)
            st.write("**Total Variable Costs (Calculated, $/Unit):**")
            st.dataframe(variable_cost_calc)

        # --- Run Button & Optimization ---
        if st.button("🚀 Run Optimization and Simulation"):
            st.header("📊 Initial Optimization Results")
            with st.spinner('Running initial optimization...'):
                random.seed(1447) # Set seed for reproducibility if needed

                # Get the actual demand column name (should be just one)
                demand_col_name = demand_aligned.columns[0]

                # Run initial optimization
                status, total_cost, y_res, x_res, fixed_c, variable_c = optimization_simulation(
                    fixed_cost_aligned, variable_cost_calc, demand_aligned, demand_col_name, capacity_aligned
                )

                if status == "Optimal" and total_cost is not None:
                    st.subheader(f"Status: {status}")
                    st.metric("Total Costs ($/Month)", f"{total_cost:,.0f}")
                    col1, col2 = st.columns(2)
                    col1.metric("Fixed Costs ($/Month)", f"{fixed_c:,.0f}")
                    col2.metric("Variable Costs ($/Month)", f"{variable_c:,.0f}")

                    # --- Display Plant Openings (Initial) ---
                    st.subheader("Plant Openings (Initial Solution)")
                    plant_locations_opt = list(demand_aligned.index) # Use aligned locations
                    plant_capacity_types_opt = list(capacity_aligned.columns) # Use aligned capacity types
                    plant_index_opt = [f'{loc}-{cap}' for loc in plant_locations_opt for cap in plant_capacity_types_opt] # Correct order

                    df_bool_initial = pd.DataFrame(
                        {'Plant Opening': [y_res.get((loc, cap), 0) for loc in plant_locations_opt for cap in plant_capacity_types_opt]},
                         index=plant_index_opt
                    )
                    df_bool_initial_int = df_bool_initial.astype(int)

                    # Bar chart using Streamlit
                    st.bar_chart(df_bool_initial_int['Plant Opening'])

                    # Display dataframe as well
                    with st.expander("View Plant Opening Data (Initial)"):
                        st.dataframe(df_bool_initial_int)

                    # --- Optional: Display Production Flows (Initial) ---
                    with st.expander("View Production Flows (Units/Month - Initial)"):
                         production_flows = {(i, j): x_res.get((i, j), 0) for i in plant_locations_opt for j in plant_locations_opt}
                         df_flows = pd.Series(production_flows).unstack(level=1).fillna(0)
                         st.dataframe(df_flows.astype(int))

                elif status == "Error":
                     st.error("Optimization failed. Please check the data inputs and ensure indices/columns match the templates.")
                else:
                    st.warning(f"Optimization finished with status: {status}. No optimal solution found (check data for inconsistencies or infeasibility).")
                    st.write(f"Objective value: {total_cost}") # Might be None or non-optimal value


            # --- Demand Simulation ---
            if status == "Optimal": # Only run simulation if initial solution is optimal
                st.header("🔄 Demand Uncertainty Simulation")
                N_SCENARIOS = 50 # Number of scenarios
                CV = 0.5 # Coefficient of Variation for demand

                with st.spinner(f'Running {N_SCENARIOS} demand scenarios...'):
                    # Prepare demand simulation dataframe
                    df_demand_sim = pd.DataFrame({'scenario': np.arange(1, N_SCENARIOS + 1)})
                    demand_base = demand_aligned.reset_index()
                    demand_col_name = demand_base.columns[1] # Assuming first col is index name, second is demand
                    index_col_name = demand_base.columns[0]

                    markets_sim = demand_base[index_col_name].values

                    for col, value in zip(markets_sim, demand_base[demand_col_name].values):
                        sigma = CV * value
                        # Ensure sigma is non-negative and finite
                        sigma = max(0, sigma) if np.isfinite(sigma) else 0
                        simulated_demand = np.random.normal(value, sigma, N_SCENARIOS)
                        df_demand_sim[col] = np.maximum(simulated_demand, 0) # Ensure demand >= 0


                    # Combine initial and simulated demand for plotting/analysis
                    df_initial_formatted = demand_aligned.T
                    df_initial_formatted['scenario'] = 0
                    df_initial_formatted = df_initial_formatted.set_index('scenario')
                    df_demand_plot = pd.concat([df_initial_formatted, df_demand_sim.set_index('scenario')])
                    df_demand_plot = df_demand_plot.astype(int)


                    # Plot simulated demand using st.line_chart
                    st.subheader("Simulated Demand Scenarios")
                    st.line_chart(df_demand_plot)
                    with st.expander("View Simulated Demand Data"):
                        st.dataframe(df_demand_plot)


                    # --- Run Simulation Loop ---
                    list_scenario_sim, list_status_sim, list_results_sim = [], [], []
                    list_fixcost_sim, list_varcost_sim, list_totald_sim = [], [], []

                    # DataFrame for simulation results (plant openings)
                    df_bool_sim = pd.DataFrame(index=plant_index_opt) # Use same index as initial df_bool

                    demand_var_sim = df_demand_sim.set_index('scenario').T # Transpose for optimization function format

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i in range(1, N_SCENARIOS + 1):
                         status_text.text(f"Running scenario {i}/{N_SCENARIOS}...")
                         status_s, objective_s, y_s, _, fixed_s, variable_s = optimization_simulation(
                             fixed_cost_aligned, variable_cost_calc, demand_var_sim, i, capacity_aligned
                         )
                         list_scenario_sim.append(i)
                         list_status_sim.append(status_s)
                         list_results_sim.append(objective_s if objective_s is not None else np.nan) # Handle non-optimal
                         list_fixcost_sim.append(fixed_s if fixed_s is not None else np.nan)
                         list_varcost_sim.append(variable_s if variable_s is not None else np.nan)
                         list_totald_sim.append(demand_var_sim[i].sum())

                         # Record plant openings for this scenario
                         df_bool_sim[i] = [y_s.get((loc, cap), 0) for loc in plant_locations_opt for cap in plant_capacity_types_opt]

                         progress_bar.progress(i / N_SCENARIOS)

                    status_text.text(f"Simulation complete ({N_SCENARIOS} scenarios).")
                    df_bool_sim = df_bool_sim.astype(int)

                    # Combine initial bool results with simulation bool results
                    df_bool_all = pd.concat([df_bool_initial_int.rename(columns={'Plant Opening': 'INITIAL'}), df_bool_sim], axis=1)

                    # Analyze unique combinations from simulation + initial
                    unique_combinations_sim = df_bool_all.T.drop_duplicates().T
                    unique_combinations_sim.columns = ['INITIAL'] + [f'com{i}' for i in range(1, len(unique_combinations_sim.columns))]

                    st.subheader("Frequency of Plant Opening Combinations")
                    com_name_sim, col_number_sim = [], []
                    for i in unique_combinations_sim.columns:
                        count = 0
                        com_name_sim.append(i)
                        for j in df_bool_all.columns:
                            # Compare columns for equality
                            if df_bool_all[j].equals(unique_combinations_sim[i]):
                                count += 1
                        col_number_sim.append(count)

                    df_com_sim = pd.DataFrame({'combination': com_name_sim, 'count': col_number_sim}).set_index('combination')

                    # Plot frequency using st.bar_chart
                    st.bar_chart(df_com_sim['count'])
                    with st.expander("View Combination Counts"):
                        st.dataframe(df_com_sim)


                    # --- Display Heatmap of Combinations ---
                    st.subheader("Heatmap of Plant Openings Across Scenarios")
                    fig_heatmap, ax_heatmap = plt.subplots(figsize=(20, 5))
                    cax = ax_heatmap.pcolor(df_bool_all, cmap='GnBu', edgecolors='k', linewidths=0.5)
                    fig_heatmap.colorbar(cax) # Add color bar using figure object
                    ax_heatmap.set_xticks(np.arange(df_bool_all.shape[1]) + 0.5)
                    ax_heatmap.set_xticklabels(df_bool_all.columns, rotation=90, fontsize=10) # Adjusted font size
                    ax_heatmap.set_yticks(np.arange(df_bool_all.shape[0]) + 0.5)
                    ax_heatmap.set_yticklabels(df_bool_all.index, fontsize=10) # Adjusted font size
                    plt.title("Plant Openings (1=Open, 0=Closed) per Scenario")
                    st.pyplot(fig_heatmap)

                    with st.expander("View Plant Opening Data (All Scenarios)"):
                        st.dataframe(df_bool_all)

                    # --- Display Summary Statistics for Costs ---
                    st.subheader("Cost Distribution Across Scenarios")
                    results_data = {
                        'Scenario': ['INITIAL'] + list_scenario_sim,
                        'Status': [status] + list_status_sim,
                        'Total Cost': [total_cost if total_cost is not None else np.nan] + list_results_sim,
                        'Fixed Cost': [fixed_c if fixed_c is not None else np.nan] + list_fixcost_sim,
                        'Variable Cost': [variable_c if variable_c is not None else np.nan] + list_varcost_sim,
                        'Total Demand': [demand_aligned[demand_col_name].sum()] + list_totald_sim
                    }
                    df_results_summary = pd.DataFrame(results_data)
                    df_results_optimal = df_results_summary[df_results_summary['Status'] == 'Optimal'].copy() # Filter for optimal

                    if not df_results_optimal.empty:
                        st.write("Summary Statistics for Optimal Scenarios:")
                        st.dataframe(df_results_optimal[['Total Cost', 'Fixed Cost', 'Variable Cost', 'Total Demand']].describe())

                        # Histogram of Total Costs
                        fig_hist, ax_hist = plt.subplots()
                        ax_hist.hist(df_results_optimal['Total Cost'].dropna(), bins=15, edgecolor='black') # dropna just in case
                        ax_hist.set_title('Distribution of Total Costs Across Optimal Scenarios')
                        ax_hist.set_xlabel('Total Cost ($/Month)')
                        ax_hist.set_ylabel('Frequency')
                        st.pyplot(fig_hist)
                    else:
                         st.warning("No optimal solutions found during the simulation.")


                    with st.expander("View Detailed Scenario Results (All Statuses)"):
                        st.dataframe(df_results_summary.set_index('Scenario'))


    except FileNotFoundError:
        st.error("One or more template files might be missing if running locally without upload.")
    except Exception as e:
        st.error(f"An error occurred loading or processing the data: {e}")
        st.exception(e) # Provides more detailed traceback

elif not use_dummy_data and not all_files_uploaded:
    st.info("👈 Please upload all 5 required Excel files using the sidebar, or toggle 'Use Dummy Data' ON.")

st.sidebar.markdown("---")
st.sidebar.write("App based on PuLP optimization.")