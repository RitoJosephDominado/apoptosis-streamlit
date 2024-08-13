import streamlit as st
import pandas as pd
import numpy as np
import time
import os

from solver import Solver

st.set_page_config(layout="wide")
steady_state_df = pd.read_csv('csvs/steady_states.csv')

if 'initial_value_df' not in st.session_state:
    st.session_state['initial_value_df'] = pd.DataFrame({
        'Species': steady_state_df.species,
        'Concentration': np.repeat(40, 13)
    })

if 'rate_df' not in st.session_state:
    st.session_state['rate_df'] = pd.read_csv('csvs/legewi_rates.txt')

col1, col2, col3 = st.columns(3)
with col1:
    st.header('Rate constants')
    rdf = st.data_editor(st.session_state.rate_df)

with col2:
    st.header('Initial Values')
    ivdf = st.data_editor(st.session_state.initial_value_df)

with col3:
    st.write('### Steady States from CRNToolBox')
    st.table(steady_state_df)

def get_sol():
    file = 'reaction_networks/legewi_wildtype.txt'
    initial_values = np.repeat(10, 13)
    initial_values = ivdf.Concentration
    rate_constants = rdf.loc[:, 'rate']
    sol = Solver(file=file, initial_values=initial_values, rate_constants=rate_constants)
    return(sol)

sol = get_sol()
st.header('Results')
print('running solve_ivp')
sol.solve(num_points=20, start_x = 0, end_x = 10)

st.pyplot(sol.plot_solution()[0])
st.table(sol.get_solution_df())