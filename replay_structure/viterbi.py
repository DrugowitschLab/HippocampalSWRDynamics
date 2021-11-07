import numpy as np


class Viterbi:
    """Implementation of the Viterbi algorithm for finding the most likely trajectory
    p(z_1:T)."""

    def __init__(self, HMM_params):
        """HMM params dictionary:
        'emission_probabilities' p(x_t|z_t) - shape (K x T)
        'transition_matrix' p(z_t|z_(t-1)) - shape (K x K)
        'initial_state_prior' p(z_t) - shape (1 x K)"""
        self.params = HMM_params
        (self.n_states, self.n_timesteps) = np.shape(
            self.params["emission_probabilities"]
        )

    def run_viterbi_algorithm(self):
        HMM_outputs = dict()
        omegas, phis = self.forward_pass()
        z_max = self.backward_pass(omegas, phis)
        HMM_outputs["omegas"] = omegas
        HMM_outputs["phis"] = phis
        HMM_outputs["z_max"] = z_max
        return HMM_outputs

    def forward_pass(self):
        # initialize alphas
        omegas = np.zeros((self.n_timesteps, self.n_states))
        phis = np.zeros((self.n_timesteps - 1, self.n_states))

        omegas[0] = np.log(self.params["initial_state_prior"]) + np.log(
            self.params["emission_probabilities"][:, 0]
        )
        # calculate recursively forward in time
        for t in range(1, self.n_timesteps):
            prediction_sum = (
                np.log(self.params["transition_matrix"][:, :]) + omegas[t - 1]
            )
            phis[t - 1] = np.argmax(prediction_sum, axis=1)
            prediction_max = np.max(prediction_sum, axis=1)
            obs_sum = (
                np.log(self.params["emission_probabilities"][:, t]) + prediction_max
            )
            omegas[t] = obs_sum
        return omegas, phis

    def backward_pass(self, omegas, phis):
        z_max = np.zeros(self.n_timesteps)
        z_max[self.n_timesteps - 1] = np.argmax(omegas[self.n_timesteps - 1])
        for t in range(self.n_timesteps - 2, -1, -1):
            z_max[t] = phis[t, int(z_max[t + 1])]
        return z_max
