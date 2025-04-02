import numpy as np
import matplotlib.pyplot as plt

class DynamicModel:
    def __init__(self, T=25, delta_S=0.5, y=10, p=1, a_sigma=-0.5, a_0=1, a_c=0.6, 
                 a_sb=0.21, a_bsigma=-0.4, beta=0.98, a_b=0.05, S_max=50, lambda_range=100, P=1000):
        self.T = T
        self.delta_S = delta_S
        self.y = y
        self.p = p
        self.a_sigma = a_sigma
        self.a_0 = a_0
        self.a_c = a_c
        self.a_sb = a_sb
        self.a_bsigma = a_bsigma
        self.beta = beta
        self.a_b = a_b
        self.P = P

        # Grids
        self.S_grid = np.linspace(0, S_max, 50)
        self.nS = len(self.S_grid)
        self.sigma_values = [0, 1]
        self.nSigma = len(self.sigma_values)
        self.lambda_grid = np.linspace(-lambda_range, lambda_range, 50)
        self.nLambda = len(self.lambda_grid)
        self.b_choices = [0, 1]

        # Value and policy arrays
        self.V = np.zeros((self.T, self.nS, self.nSigma, self.nLambda))
        self.Policy = np.zeros((self.T, self.nS, self.nSigma, self.nLambda), dtype=int)
    
    def utility(self, S, sigma, b):
        return np.exp(self.a_sigma * sigma) * (self.a_0 + (self.y - self.p * b) ** self.a_c *
                                               (1 + b) ** self.a_b * (1 + S * b) ** self.a_sb *
                                               (1 + sigma * b) ** self.a_bsigma)
    
    def transition_S(self, S, b):
        new_S = (1 - self.delta_S) * S + b
        return max(self.S_grid[0], min(new_S, self.S_grid[-1]))
    
    def pi_of_S(self, S, k=0.1, S0=None):
        if S0 is None:
            S0 = self.S_grid[-1] / 2
        return 1.0 / (1.0 + np.exp(-k * (S - S0)))
    
    def transition_lambda(self, lambda_val, S, sigma_lagged, sigma_current):
        if sigma_lagged == 0:
            if sigma_current == 1:
                return lambda_val + np.log(self.pi_of_S(S) / self.pi_of_S(0))
            else:
                return lambda_val + np.log((1 - self.pi_of_S(S)) / (1 - self.pi_of_S(0)))
        return lambda_val
    
    def prob_sigma_next_is_one(self, sigma_current, lambda_val, S):
        if sigma_current == 1:
            return 1.0
        num = self.pi_of_S(S) * np.exp(lambda_val) + (self.pi_of_S(0) if S != 0 else 0)
        return num / (1.0 + np.exp(lambda_val))
    
    def backward_induction(self):
        # Last period
        t = self.T - 1
        for i, S_val in enumerate(self.S_grid):
            for j, sigma_val in enumerate(self.sigma_values):
                for k, lambda_val in enumerate(self.lambda_grid):
                    best_value = -np.inf
                    best_choice = 0
                    for b in self.b_choices:
                        u = self.utility(S_val, sigma_val, b)
                        if u > best_value:
                            best_value = u
                            best_choice = b
                    self.V[t, i, j, k] = best_value
                    self.Policy[t, i, j, k] = best_choice
        
        # Earlier periods

        for t in reversed(range(self.T - 1)):
            for i, S_val in enumerate(self.S_grid):
                for j, sigma_val in enumerate(self.sigma_values):
                    for k, lambda_val in enumerate(self.lambda_grid):
                        best_value = -np.inf
                        best_choice = 0
                        for b in self.b_choices:
                            u = self.utility(S_val, sigma_val, b)
                            S_next = self.transition_S(S_val, b)
                            i_next = np.argmin(np.abs(self.S_grid - S_next))
                            p1 = self.prob_sigma_next_is_one(sigma_val, lambda_val, S_val)
                            lambda_next_1 = self.transition_lambda(lambda_val, S_val, sigma_val, 1)
                            lambda_next_0 = self.transition_lambda(lambda_val, S_val, sigma_val, 0)
                            k_next_1 = np.argmin(np.abs(self.lambda_grid - lambda_next_1))
                            k_next_0 = np.argmin(np.abs(self.lambda_grid - lambda_next_0))
                            cont_val = (p1 * self.V[t + 1, i_next, 1, k_next_1] +
                                        (1 - p1) * self.V[t + 1, i_next, 0, k_next_0])
                            total_value = u + self.beta * cont_val
                            if total_value > best_value:
                                best_value = total_value
                                best_choice = b
                        self.V[t, i, j, k] = best_value
                        self.Policy[t, i, j, k] = best_choice


    def bi_vec(self):
        print('hej3')

        

        # Assume the following are defined:
        # self.S_grid, self.sigma_values, self.lambda_grid, self.b_choices, self.T, self.beta, self.utility, self.transition_S, self.prob_sigma_next_is_one, self.transition_lambda, self.V, self.Policy

        # Create the meshgrid for S, sigma, and lambda
        S_mesh, sigma_mesh, lambda_mesh = np.meshgrid(self.S_grid, self.sigma_values, self.lambda_grid, indexing='ij')

        # Flatten meshgrid to apply functions vectorially
        S_flat, sigma_flat, lambda_flat = S_mesh.ravel(), sigma_mesh.ravel(), lambda_mesh.ravel()

        # Loop over time steps in reversed order
        for t in reversed(range(self.T - 1)):
            # Compute utility for all combinations of (S, sigma, b) at once
            u_vals = np.array([[self.utility(S, sigma, b) for b in self.b_choices] for S, sigma in zip(S_flat, sigma_flat)])

            # Compute state transitions for all b choices
            S_next_vals = np.array([[self.transition_S(S, b) for b in self.b_choices] for S in S_flat])

            # Find the closest indices in the state grid
            i_next_vals = np.abs(self.S_grid[:, None] - S_next_vals.T).argmin(axis=0)  # Ensures correct broadcasting

            # Compute transition probabilities for sigma
            p1_vals = np.array([self.prob_sigma_next_is_one(sigma, lambda_val, S) for sigma, lambda_val, S in zip(sigma_flat, lambda_flat, S_flat)])

            # **Fix: Compute lambda transitions correctly across b choices**
            lambda_next_1_vals = np.array([[self.transition_lambda(lambda_val, S, sigma, 1) for b in self.b_choices] for lambda_val, S, sigma in zip(lambda_flat, S_flat, sigma_flat)])
            lambda_next_0_vals = np.array([[self.transition_lambda(lambda_val, S, sigma, 0) for b in self.b_choices] for lambda_val, S, sigma in zip(lambda_flat, S_flat, sigma_flat)])

            # Find the nearest lambda indices
            k_next_1_vals = np.abs(self.lambda_grid[:, None] - lambda_next_1_vals.T).argmin(axis=0)
            k_next_0_vals = np.abs(self.lambda_grid[:, None] - lambda_next_0_vals.T).argmin(axis=0)

            # **Ensure Correct Broadcasting for Continuation Values**
            cont_vals = (p1_vals[:, None] * self.V[t + 1, i_next_vals, 1, k_next_1_vals] +
                        (1 - p1_vals[:, None]) * self.V[t + 1, i_next_vals, 0, k_next_0_vals])

            # Compute total values
            total_vals = u_vals + self.beta * cont_vals

            # Get best choices
            best_choice_indices = np.argmax(total_vals, axis=1)
            best_value = np.max(total_vals, axis=1)

            # Reshape to match meshgrid dimensions
            self.V[t, :, :, :] = best_value.reshape(S_mesh.shape)
            self.Policy[t, :, :, :] = np.array([self.b_choices[i] for i in best_choice_indices]).reshape(S_mesh.shape)





    def simulate_and_plot(self, plot=1):
        import matplotlib.pyplot as plt
        np.random.seed(42)  # For reproducibility

        # Initialize simulation arrays
        S_sim = np.zeros((self.T, self.P))
        sigma_sim = np.zeros((self.T, self.P), dtype=int)
        lambda_sim = np.random.choice(self.lambda_grid, size=(self.T, self.P))  # Ensure 2D
        b_sim = np.zeros((self.T, self.P), dtype=int)
        
        # Simulate forward in time
        for t in range(self.T - 1):
            for person in range(self.P):
                i_idx = np.argmin(np.abs(self.S_grid - S_sim[t, person]))
                j_idx = sigma_sim[t, person]

                k_idx = np.argmin(np.abs(self.lambda_grid - lambda_sim[t, person]))  # Ensure valid indexing

                # Retrieve optimal decision
                b_sim[t, person] = self.Policy[t, i_idx, j_idx, k_idx]

                # Transition states
                S_sim[t+1, person] = self.transition_S(S_sim[t, person], b_sim[t, person])

                # Update sigma
                if sigma_sim[t, person] == 0:
                    sigma_sim[t+1, person] = 1 if np.random.rand() < self.pi_of_S(S_sim[t, person]) else 0
                else:
                    sigma_sim[t+1, person] = 1  # If already sick, stays sick

                # Transition lambda
                lambda_sim[t+1, person] = self.transition_lambda(
                    lambda_sim[t, person], S_sim[t, person], sigma_sim[t, person], sigma_sim[t+1, person]
                )

        if plot==1:
            # For example, plot the fraction of persons smoking over time.
            fraction_smoking = np.mean(b_sim, axis=1)

            plt.figure(figsize=(8, 6))
            plt.plot(range(self.T), fraction_smoking, marker='o')
            plt.xlabel('Time Period')
            plt.ylabel('Fraction Smoking')
            plt.title('Simulated Fraction of Persons Who Smoke Over Time')
            plt.grid(True)
            plt.show()

            # Also, we can plot average lambda over time.
            avg_lambda = np.mean(lambda_sim, axis=1)
            plt.figure(figsize=(8, 6))
            plt.plot(range(self.T), avg_lambda, marker='o', color='orange')
            plt.xlabel('Time Period')
            plt.ylabel('Average lambda')
            plt.title('Average Belief (lambda) Over Time')
            plt.grid(True)
            plt.show()

            # Compute fraction of sick persons (sigma == 1) over time.
            fraction_sick = np.mean(sigma_sim, axis=1)

            plt.figure(figsize=(8, 6))
            plt.plot(range(self.T), fraction_sick, marker='o', color='red')
            plt.xlabel('Time Period')
            plt.ylabel('Fraction Sick (σ=1)')
            plt.title('Simulated Fraction of Persons Who Are Sick Over Time')
            plt.grid(True)
            plt.show()
        return S_sim, sigma_sim, lambda_sim, b_sim

