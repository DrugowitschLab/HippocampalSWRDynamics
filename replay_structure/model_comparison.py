import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal
from scipy.special import logsumexp
from typing import Dict, Union

from replay_structure.metadata import MODELS_AS_STR
from replay_structure.structure_models_gridsearch import Structure_Gridsearch
from replay_structure.utils import InvGamma_Distribution, LogNorm2D_Distribution


class Gridsearch_Marginalization:
    """Marginalizes data likelihood over paramter gridsearch,
    p(x_1:T|M) = Σ_ϴ p(x_1:T|ϴ, M)p(ϴ|M).
    """

    def __init__(
        self,
        gridsearch: Structure_Gridsearch,
        marginalization_info=None,
        exclude_param_edges: bool = True,
    ):
        self.gridsearch_params: dict = gridsearch.gridsearch_params
        self.gridsearch_results: np.ndarray = gridsearch.gridsearch_results
        self.exclude_param_edges = exclude_param_edges
        if len(self.gridsearch_results.shape) == 2:
            self.fit_2d_prior = False
        else:
            self.fit_2d_prior = True
        if marginalization_info is None:
            self.marginalization_info = self.get_marginalization_info(
                self.gridsearch_params, self.gridsearch_results
            )
        else:
            self.marginalization_info = marginalization_info
            self.simulated_marginalization_info = self.get_marginalization_info(
                self.gridsearch_params, self.gridsearch_results, fit_prior=False
            )
        self.marginalized_model_evidences = self.marginalize_model_evidences(
            self.marginalization_info["fit_prior_evaluated_at_gridsearch_params"]
        )

    def get_marginalization_info(
        self,
        gridsearch_params: dict,
        gridsearch_results: np.ndarray,
        fit_prior: bool = True,
    ):
        marginalization_info = dict()
        (
            marginalization_info["best_fit_gridsearch_params"],
            marginalization_info["argmax_mat"],
        ) = self.get_best_fit_gridsearch_params(
            self.gridsearch_params, self.gridsearch_results
        )
        if fit_prior:
            marginalization_info["fit_prior_params"], marginalization_info[
                "fit_prior_evaluated_at_gridsearch_params"
            ] = self.fit_prior(marginalization_info["best_fit_gridsearch_params"])
        return marginalization_info

    def get_best_fit_gridsearch_params(
        self, gridsearch_params: dict, gridsearch_results: np.ndarray
    ) -> tuple:
        best_fit_params = dict()
        if len(gridsearch_results.shape) == 2:
            argmax_mat = gridsearch_results.T == np.nanmax(gridsearch_results, axis=1)
            duplicate_max = np.argwhere(np.sum(argmax_mat, axis=0) > 1)
            if len(duplicate_max) > 0:
                for ripple in duplicate_max:
                    sd_ind = np.where(np.squeeze(argmax_mat[:, ripple]))
                    argmax_mat[:, ripple] = False
                    argmax_mat[sd_ind[0][0], ripple] = True
            sd_inds = np.where(argmax_mat.T)[1]
            best_fit_params["sd_meters"] = gridsearch_params["sd_array_meters"][sd_inds]
        elif len(gridsearch_results.shape) == 3:
            argmax_mat = gridsearch_results.T == np.nanmax(
                gridsearch_results, axis=(1, 2)
            )
            duplicate_max = np.argwhere(np.sum(argmax_mat, axis=(0, 1)) > 1)
            if len(duplicate_max) > 0:
                for ripple in duplicate_max:
                    sd_ind, decay_ind = np.where(np.squeeze(argmax_mat[:, :, ripple]))
                    argmax_mat[:, :, ripple] = False
                    argmax_mat[sd_ind[0], decay_ind[0], ripple] = True
            sd_inds = np.where(argmax_mat.T)[1]
            decay_inds = np.where(argmax_mat.T)[2]
            best_fit_params["sd_meters"] = gridsearch_params["sd_array_meters"][sd_inds]
            best_fit_params["decay"] = gridsearch_params["decay_array"][decay_inds]
        else:
            raise Exception("Unexpected gridsearch results dimensions")
        # test for multiple argmax
        print(argmax_mat.sum())
        if argmax_mat.sum() > self.gridsearch_results.shape[0]:
            print("Argmax Duplicates")
        return (best_fit_params, argmax_mat)

    def fit_prior(self, best_fit_gridsearch_params: dict) -> tuple:
        fit_prior_params: Dict[
            str, Union[LogNorm2D_Distribution, InvGamma_Distribution]
        ] = dict()
        fit_prior_evaluated_at_gridsearch_params = dict()
        if self.fit_2d_prior:
            fit_prior_params["2d_lognorm"] = self.fit_multivariate_normal_prior(
                best_fit_gridsearch_params
            )
            assert isinstance(fit_prior_params["2d_lognorm"], LogNorm2D_Distribution)
            fit_prior_evaluated_at_gridsearch_params[
                "2d_lognorm"
            ] = self.calc_multivariate_lognormal_prior(fit_prior_params["2d_lognorm"])
        else:
            for param in best_fit_gridsearch_params:
                if param == "sd_meters":
                    fit_prior_params[param] = self.fit_sd_prior(
                        best_fit_gridsearch_params[param]
                    )
                    fit_prior_evaluated_at_gridsearch_params[
                        param
                    ] = self.calc_sd_prior(fit_prior_params[param])
                elif param == "decay":
                    fit_prior_params[param] = self.fit_decay_prior(
                        best_fit_gridsearch_params[param]
                    )
                    assert isinstance(fit_prior_params[param], InvGamma_Distribution)
                    fit_prior_evaluated_at_gridsearch_params[
                        param
                    ] = self.calc_decay_prior(fit_prior_params[param])
        return fit_prior_params, fit_prior_evaluated_at_gridsearch_params

    def fit_sd_prior(self, best_fit_sd_array: np.ndarray) -> InvGamma_Distribution:
        if self.exclude_param_edges:
            best_fit_sd_array = best_fit_sd_array[
                (best_fit_sd_array != self.gridsearch_params["sd_array_meters"][0])
                & (best_fit_sd_array != self.gridsearch_params["sd_array_meters"][-1])
            ]
        a, loc, scale = invgamma.fit(best_fit_sd_array)
        return InvGamma_Distribution(a=a, loc=loc, scale=scale)

    def fit_decay_prior(
        self, best_fit_decay_array: np.ndarray
    ) -> InvGamma_Distribution:
        if self.exclude_param_edges:
            best_fit_decay_array = best_fit_decay_array[
                (best_fit_decay_array != self.gridsearch_params["decay_array"][0])
                & (best_fit_decay_array != self.gridsearch_params["decay_array"][-1])
            ]
        a, loc, scale = invgamma.fit(best_fit_decay_array)
        return InvGamma_Distribution(a=a, loc=loc, scale=scale)

    def fit_multivariate_normal_prior(
        self, best_fit_gridsearch_params: dict
    ) -> LogNorm2D_Distribution:

        sd_edges = (
            best_fit_gridsearch_params["sd_meters"]
            == self.gridsearch_params["sd_array_meters"][0]
        ) | (
            best_fit_gridsearch_params["sd_meters"]
            == self.gridsearch_params["sd_array_meters"][-1]
        )
        decay_edges = (
            best_fit_gridsearch_params["decay"]
            == self.gridsearch_params["decay_array"][0]
        ) | (
            best_fit_gridsearch_params["decay"]
            == self.gridsearch_params["decay_array"][-1]
        )
        not_edge = ~sd_edges * ~decay_edges
        best_fit_sd_array_edges_removed = best_fit_gridsearch_params["sd_meters"][
            not_edge
        ]
        best_fit_decay_array_edges_removed = best_fit_gridsearch_params["decay"][
            not_edge
        ]
        data = np.vstack(
            (
                np.log(best_fit_sd_array_edges_removed),
                np.log(best_fit_decay_array_edges_removed),
            )
        )
        mean = np.mean(data, axis=1)
        cov = np.cov(data)

        return LogNorm2D_Distribution(mean=mean, cov=cov)

    def calc_sd_prior(self, fit_sd_prior: InvGamma_Distribution) -> np.ndarray:
        prior = invgamma.pdf(
            self.gridsearch_params["sd_array_meters"],
            a=fit_sd_prior.a,
            loc=fit_sd_prior.loc,
            scale=fit_sd_prior.scale,
        )
        return prior / np.sum(prior)

    def calc_decay_prior(self, fit_decay_prior: InvGamma_Distribution) -> np.ndarray:
        prior = invgamma.pdf(
            self.gridsearch_params["decay_array"],
            a=fit_decay_prior.a,
            loc=fit_decay_prior.loc,
            scale=fit_decay_prior.scale,
        )
        return prior / np.sum(prior)

    def calc_multivariate_lognormal_prior(
        self, fit_2dnormal_prior: LogNorm2D_Distribution
    ) -> np.ndarray:
        x = self.gridsearch_params["decay_array"]
        y = self.gridsearch_params["sd_array_meters"]
        xx, yy = np.meshgrid(x, y)
        prior = multivariate_normal.pdf(
            np.dstack((np.log(yy), np.log(xx))),
            mean=fit_2dnormal_prior.mean,
            cov=fit_2dnormal_prior.cov,
        )
        return prior / np.sum(prior)

    def marginalize_model_evidences(
        self, fit_prior_evaluated_at_gridsearch_params: dict
    ):
        if self.fit_2d_prior:
            prior_mesh = fit_prior_evaluated_at_gridsearch_params["2d_lognorm"]
            marginalized_model_evidences = logsumexp(
                self.gridsearch_results + np.log(prior_mesh), axis=(1, 2)
            )
        else:
            if len(fit_prior_evaluated_at_gridsearch_params) == 1:
                sd_prior_normalized = (
                    fit_prior_evaluated_at_gridsearch_params["sd_meters"]
                    / fit_prior_evaluated_at_gridsearch_params["sd_meters"].sum()
                )
                marginalized_model_evidences = logsumexp(
                    self.gridsearch_results + np.log(sd_prior_normalized), axis=1
                )
            elif len(fit_prior_evaluated_at_gridsearch_params) == 2:
                sd_prior_normalized = (
                    fit_prior_evaluated_at_gridsearch_params["sd_meters"]
                    / fit_prior_evaluated_at_gridsearch_params["sd_meters"].sum()
                )
                decay_prior_normalized = (
                    fit_prior_evaluated_at_gridsearch_params["decay"]
                    / fit_prior_evaluated_at_gridsearch_params["decay"].sum()
                )
                decay_prior_mesh, sd_prior_mesh = np.meshgrid(
                    decay_prior_normalized, sd_prior_normalized
                )
                marginalized_model_evidences = logsumexp(
                    self.gridsearch_results
                    + np.log(sd_prior_mesh)
                    + np.log(decay_prior_mesh),
                    axis=(1, 2),
                )
        return marginalized_model_evidences


class Model_Comparison:
    """Performs model comparison across structure models per SWR to identify the best
    fit model, and performs random effects analysis to infer the distribution of models
    across SWRs within a session.
    """

    def __init__(self, model_evidences: dict, random_effects_prior: int = 5):
        self.random_effects_prior = random_effects_prior
        self.results_dataframe = self.make_results_dataframe(model_evidences)
        self.max_ll_counts = self.results_dataframe["mll_model"][
            ~np.any(np.isnan(self.results_dataframe[MODELS_AS_STR].values), axis=1)
        ].value_counts()
        for model in MODELS_AS_STR:
            if model not in self.max_ll_counts:
                self.max_ll_counts[model] = 0
        self.random_effects_results = self.run_random_effects(
            self.results_dataframe[MODELS_AS_STR]
        )

    def make_results_dataframe(self, model_evidences: dict):
        columns = MODELS_AS_STR
        results_dataframe = pd.DataFrame(columns=columns)
        for model in MODELS_AS_STR:
            results_dataframe[model] = model_evidences[model]
        results_dataframe["mll_model"] = results_dataframe[MODELS_AS_STR].idxmax(axis=1)
        print(results_dataframe.head())
        print(
            results_dataframe["mll_model"][
                ~np.any(np.isnan(results_dataframe[MODELS_AS_STR].values), axis=1)
            ].value_counts()
        )
        return results_dataframe

    def run_random_effects(self, results_dataframe, n_iterations=500, burnin=50):
        np.random.seed(0)
        results_nonan = results_dataframe.values[
            ~np.any(np.isnan(results_dataframe.values), axis=1)
        ]
        results_nonan = (results_nonan.T - results_nonan.max(axis=1)).T
        n_spikemats, n_models = results_nonan.shape

        gibbs_results = np.zeros((n_iterations, n_models))
        alpha_m_all = np.zeros((n_iterations, n_models))

        print(n_spikemats)
        alpha_m = np.ones(n_models) * self.random_effects_prior
        for n in range(n_iterations):
            r_m = np.random.dirichlet(alpha_m, size=1)
            gibbs_results[n] = r_m

            u_nm = (results_nonan + np.log(r_m)).T
            u_nm = u_nm - np.max(u_nm, axis=0)
            u_nm = np.exp(u_nm)
            g_nm = (u_nm / np.sum(u_nm, axis=0)).T

            a_n = np.zeros((n_spikemats, n_models))
            for i in range(n_spikemats):
                a_n[i] = np.random.multinomial(1, g_nm[i])
            beta_m = np.sum(a_n, axis=0).astype(int)
            alpha_m = self.random_effects_prior + beta_m

            alpha_m_all[n] = alpha_m
            p_models = np.mean(gibbs_results[burnin:], axis=0)
            p_exceedance = np.mean(
                (gibbs_results[burnin:].T == np.max(gibbs_results[burnin:], axis=1)).T,
                axis=0,
            )

        print(p_models)
        random_effects_results = {
            "gibbs": gibbs_results,
            "alpha_m": alpha_m_all,
            "p_models": p_models,
            "p_exceedance": p_exceedance,
        }
        return random_effects_results


class Factorial_Model_Comparison:
    """Performs model comparison across structure models and emission models per SWR to
    identify the best fit dynamics/emission model, and performs factorial random effects
    analysis to infer the distribution of dynamics/emission models across SWRs within a
    session. For Figure S1kl.
    """

    def __init__(
        self, model_evidences: Dict[str, dict], random_effects_prior: int = 15
    ):
        self.random_effects_prior = random_effects_prior
        self.results_dataframe = self.make_results_dataframe(model_evidences)
        self.random_effects_results = self.run_factorial_random_effects(
            self.results_dataframe
        )

    def make_results_dataframe(self, model_evidences: dict):
        columns = [
            f"{model}_{likelihood}"
            for model in MODELS_AS_STR
            for likelihood in ["poisson", "negbinomial"]
        ]
        results_dataframe = pd.DataFrame(columns=columns)
        for model in MODELS_AS_STR:
            for likelihood in ["poisson", "negbinomial"]:
                results_dataframe[f"{model}_{likelihood}"] = model_evidences[
                    likelihood
                ][model]
        return results_dataframe

    def run_factorial_random_effects(
        self, results_dataframe, n_iterations=500, burnin=50
    ):
        np.random.seed(0)
        results_nonan = results_dataframe.values[
            ~np.any(np.isnan(results_dataframe.values), axis=1)
        ]
        results_nonan = (results_nonan.T - results_nonan.max(axis=1)).T
        n_spikemats, n_models = results_nonan.shape

        gibbs_results = np.zeros((n_iterations, n_models))
        alpha_m_all = np.zeros((n_iterations, n_models))

        print(n_spikemats)
        alpha_m = np.ones(n_models) * self.random_effects_prior

        for n in range(n_iterations):
            r_m = np.random.dirichlet(alpha_m, size=1)
            gibbs_results[n] = r_m

            u_nm = (results_nonan + np.log(r_m)).T
            u_nm = u_nm - np.max(u_nm, axis=0)
            u_nm = np.exp(u_nm)
            g_nm = (u_nm / np.sum(u_nm, axis=0)).T

            a_n = np.zeros((n_spikemats, n_models))
            for i in range(n_spikemats):
                a_n[i] = np.random.multinomial(1, g_nm[i])
            beta_m = np.sum(a_n, axis=0).astype(int)
            alpha_m = self.random_effects_prior + beta_m

            alpha_m_all[n] = alpha_m

        p_models = np.mean(gibbs_results[burnin:], axis=0)
        p_exceedance_all = np.mean(
            (gibbs_results[burnin:].T == np.max(gibbs_results[burnin:], axis=1)).T,
            axis=0,
        )

        p_emission_models, p_exceedance_emissions = self.collapse_over_dynamics_models(
            alpha_m_all[burnin:]
        )
        p_dynamics_models, p_exceedance_dynamics = self.collapse_over_emission_models(
            alpha_m_all[burnin:]
        )

        print(p_models)
        random_effects_results = {
            "gibbs": gibbs_results,
            "alpha_m": alpha_m_all,
            "p_all_models": p_models,
            "p_dynamics_models": p_dynamics_models,
            "p_emission_models": p_emission_models,
            "p_exceedance_all": p_exceedance_all,
            "p_exceedance_dynamics": p_exceedance_dynamics,
            "p_exceedance_emissions": p_exceedance_emissions,
        }
        return random_effects_results

    @staticmethod
    def collapse_over_dynamics_models(alpha_m_all: np.ndarray) -> np.ndarray:
        a_sum = alpha_m_all.sum(axis=0)
        emissions_sum = np.array([a_sum[i::2].sum() for i in range(2)])
        p_models = emissions_sum / emissions_sum.sum()

        alpha_collapsed = np.zeros((alpha_m_all.shape[0], 2))
        for i in range(2):
            alpha_collapsed[:, i] = alpha_m_all[:, i::2].sum(axis=1)
        p_exceedance = np.mean(
            (alpha_collapsed.T == np.max(alpha_collapsed, axis=1)).T, axis=0
        )
        return p_models, p_exceedance

    @staticmethod
    def collapse_over_emission_models(alpha_m_all: np.ndarray) -> np.ndarray:
        a_sum = alpha_m_all.sum(axis=0)
        models_sum = np.array([a_sum[i * 2 : i * 2 + 2].sum() for i in range(5)])
        p_models = models_sum / models_sum.sum()

        alpha_collapsed = np.zeros((alpha_m_all.shape[0], 5))
        for i in range(5):
            alpha_collapsed[:, i] = alpha_m_all[:, i * 2 : i * 2 + 2].sum(axis=1)
        p_exceedance = np.mean(
            (alpha_collapsed.T == np.max(alpha_collapsed, axis=1)).T, axis=0
        )
        return p_models, p_exceedance
