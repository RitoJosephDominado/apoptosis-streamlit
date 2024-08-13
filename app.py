import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import psutil

from solver import Solver

steady_state_df = pd.read_csv('csvs/steady_states.csv')

if 'initial_value_df' not in st.session_state:
    st.session_state['initial_value_df'] = pd.DataFrame({
        'Species': steady_state_df.species,
        'Concentration': np.repeat(40, 13)
    })

if 'rate_df' not in st.session_state:
    st.session_state['rate_df'] = pd.read_csv('csvs/legewi_rates.txt')


exit_app = st.sidebar.button("Shut Down")
if exit_app:
    pid = os.getpid()
    p = psutil.Process(pid)
    p.terminate()

col1, col2 = st.columns(2)
with col1:
    st.header('Rate constants')
    # st.data_editor(pd.read_csv('reaction_networks/legewi_rates.txt'))
    rdf = st.data_editor(st.session_state.rate_df)


with col2:
    st.header('Initial Values')
    ivdf = st.data_editor(st.session_state.initial_value_df)

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
sol.solve(num_points=50, start_x = 0, end_x = 10)

st.pyplot(sol.plot_solution()[0])
st.table(sol.get_solution_df())