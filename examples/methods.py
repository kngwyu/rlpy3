from rlpy.Agents import LSPI, NaturalActorCritic, Q_Learning, SARSA
from rlpy.Policies import eGreedy, GibbsPolicy
from rlpy.Representations import (
    gaussian_kernel,
    iFDD,
    iFDDK,
    IndependentDiscretization,
    KernelizediFDD,
    RBF,
    Tabular,
)


def tabular_lspi(domain, max_steps, discretization=20):
    tabular = Tabular(domain, discretization=discretization)
    policy = eGreedy(tabular, epsilon=0.1)
    return LSPI(policy, tabular, domain.discount_factor, max_steps, 1000)


def tabular_nac(
    domain, discretization=20, forgetting_rate=0.3, lambda_=0.7, learn_rate=0.1
):
    tabular = Tabular(domain, discretization=discretization)
    return NaturalActorCritic(
        GibbsPolicy(tabular),
        tabular,
        forgetting_rate=forgetting_rate,
        min_steps_between_updates=100,
        max_steps_between_updates=1000,
        lambda_=lambda_,
        learn_rate=learn_rate,
    )


def tabular_q(
    domain, discretization=20, lambda_=0.3, initial_learn_rate=0.1, boyan_N0=100
):
    tabular = Tabular(domain, discretization=discretization)
    return Q_Learning(
        eGreedy(tabular, epsilon=0.1),
        tabular,
        discount_factor=domain.discount_factor,
        lambda_=lambda_,
        initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan",
        boyan_N0=boyan_N0,
    )


def tabular_sarsa(domain, discretization=20, lambda_=0.3):
    tabular = Tabular(domain, discretization=discretization)
    policy = eGreedy(tabular, epsilon=0.1)
    return SARSA(policy, tabular, domain.discount_factor, lambda_=lambda_)


def ifdd_q(
    domain,
    discretization=20,
    threshold=1.0,
    lambda_=0.3,
    initial_learn_rate=0.1,
    boyan_N0=100,
):
    ifdd = iFDD(
        domain,
        discovery_threshold=threshold,
        initial_representation=IndependentDiscretization(
            domain, discretization=discretization
        ),
        useCache=True,
        iFDDPlus=True,
    )
    return Q_Learning(
        eGreedy(ifdd, epsilon=0.1),
        ifdd,
        discount_factor=domain.discount_factor,
        lambda_=lambda_,
        initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan",
        boyan_N0=boyan_N0,
    )


def ifddk_q(
    domain,
    discretization=20,
    threshold=1.0,
    lambda_=0.3,
    initial_learn_rate=0.1,
    boyan_N0=100,
):
    ifddk = iFDDK(
        domain,
        discovery_threshold=threshold,
        initial_representation=IndependentDiscretization(
            domain, discretization=discretization
        ),
        sparsify=True,
        useCache=True,
        lazy=True,
        lambda_=lambda_,
    )
    return Q_Learning(
        eGreedy(ifddk, epsilon=0.1),
        ifddk,
        discount_factor=domain.discount_factor,
        lambda_=lambda_,
        initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan",
        boyan_N0=boyan_N0,
    )


def kifdd_q(
    domain,
    kernel_resolution,
    threshold=1.0,
    lambda_=0.3,
    initial_learn_rate=0.1,
    boyan_N0=100,
):
    kernel_width = (
        domain.statespace_limits[:, 1] - domain.statespace_limits[:, 0]
    ) / kernel_resolution
    kifdd = KernelizediFDD(
        domain,
        sparsify=True,
        kernel=gaussian_kernel,
        kernel_args=[kernel_width],
        active_threshold=0.01,
        discover_threshold=threshold,
        normalization=True,
        max_active_base_feat=10,
        max_base_feat_sim=0.5,
    )
    return Q_Learning(
        eGreedy(kifdd, epsilon=0.1),
        kifdd,
        discount_factor=domain.discount_factor,
        lambda_=lambda_,
        initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan",
        boyan_N0=boyan_N0,
    )


def rbf_q(
    domain,
    seed,
    num_rbfs=96,
    resolution=21,
    initial_learn_rate=0.1,
    lambda_=0.3,
    boyan_N0=100,
):
    rbf = RBF(
        domain,
        num_rbfs=num_rbfs,
        resolution_max=resolution,
        resolution_min=resolution,
        const_feature=False,
        normalize=True,
        seed=seed,
    )
    return Q_Learning(
        eGreedy(rbf, epsilon=0.1),
        rbf,
        discount_factor=domain.discount_factor,
        lambda_=lambda_,
        initial_learn_rate=initial_learn_rate,
        learn_rate_decay_mode="boyan",
        boyan_N0=boyan_N0,
    )
