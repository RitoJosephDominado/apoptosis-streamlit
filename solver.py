import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import re

import time

class Solver:
    def __init__(self, file, initial_values, rate_constants):
        self.file = file
        self.initial_values = initial_values
        self.rate_constants = rate_constants
        self.reaction_list, self.species_list = self.parse_reaction_file(file)
        # self.ode_func = self.generate_ode_func()

    
    def parse_reaction_file(self, file):
        with open(file, 'r') as f:
            reaction_text_list = f.readlines()
            reaction_list = []
            species_set = set()

        for reaction_text in reaction_text_list:
            x = reaction_text.split(':')[1]
            reactants, products = x.strip().split('=>')
            reactants = [r.strip() for r in reactants.split('+') if r.strip()]
            products = [p.strip() for p in products.split('+') if p.strip()]

            reaction_list.append((reactants, products))
            species_set.update(reactants)
            species_set.update(products)
        species_list = sorted(list(species_set))
        return(reaction_list, species_list)
    
    def generate_ode_func(self):
        species_list = self.species_list
        reaction_list = self.reaction_list
        def odes(t, y, k):
            dydt = np.zeros(len(species_list))
            species_index = {sp: i for i, sp in enumerate(species_list)}

            for i, (reactants, products) in enumerate(reaction_list):
                if not reactants:
                    rate = k[i]
                else:
                    rate = k[i]
                    for reactant in reactants:
                        rate *= y[species_index[reactant]]
                    for reactant in reactants:
                        dydt[species_index[reactant]] -= rate
                for product in products:
                    dydt[species_index[product]] += rate
            return(dydt)
        return(odes)
    
    def solve(self, num_points, start_x=0, end_x=50):
        f = self.generate_ode_func()
        start_time = time.time()
        if not end_x:
            end_x = num_points
        t_span = (start_x, end_x)
        points_to_evaluate = np.linspace(t_span[0], t_span[1], num_points)
        # solution = solve_ivp(self.ode_func, t_span, self.initial_values, t_eval=points_to_evaluate, args = self.rate_constants)
        solution = solve_ivp(f, t_span, self.initial_values, t_eval=points_to_evaluate, args = (self.rate_constants,))
        end_time = time.time()

        print(f'Elapsed time: {end_time - start_time}')
        self.solution = solution
        return(solution)
    
    def get_solution_df(self):
        df = pd.DataFrame(dict(zip(self.species_list, self.solution.y)))
        return(df)
    
    def plot_solution(self):
        fig, ax = plt.subplots(figsize=(15, 8))
        for i, sp in enumerate(self.species_list):
            ax.plot(self.solution.t, self.solution.y[i], label=f'[{sp}](t)', marker='o')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.set_title('Concentration vs. Time for the Reaction Network')
        ax.legend()
        ax.grid(True)
        return(fig, ax)
