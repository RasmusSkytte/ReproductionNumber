import numpy as np
import pandas as pd
import streamlit as st
import datetime
import os
from scipy.interpolate import interp1d
from scipy.optimize    import minimize_scalar
from scipy.linalg      import eig

def error_propergation_multiplication(*args) :

    # Iterate to determine dimension
    dim = np.max([np.size(arg[0]) for arg in args])

    # Start
    f  = np.ones(dim)
    df = np.zeros(dim)

    # Iterate
    for p in args :

        # Check if input has enable/disable flag
        if len(p) == 3 :
            v, dv, flag = p

            if not flag :
                continue

        else :
            v, dv = p

        f *= v

        I = dv > 0
        if np.size(v) > 1 :
            df[I] += np.power(dv[I] / v[I], 2.0)
        elif I :
            df += np.power(dv / v, 2.0)

    df = np.abs(f) * np.sqrt(df)

    return f, df


def error_propergation_addition(*args) :

    # Start
    f  = 0
    df = 0

    # Iterate
    for p in args :
        v, dv = p

        f  += v

        df += np.power(dv, 2.0)

    df = np.sqrt(df)

    return f, df


def temperature_model(start_month, ref_month) :

    # Average of maxiumum tempetatures   (jan, feb, mar,  apr,  maj,  jun,  jul,  aug,  sep,  okt, nov, dec)
    average_max_temperatur_dk = np.array([3.6, 3.7, 6.4, 11.2, 15.6, 18.5, 21.2, 21.2, 17.2, 12.3, 7.6, 4.7])
    # source: https://www.dmi.dk/vejrarkiv/normaler-danmark/

    # Model fit
    p = {'A' : 0.33286897, 'K' : 0.07675696, 'B' : 0.12298820, 'x0' : 10.53276565, 'sc' : 0.058347728, 'nu' : 0.99193751}
    beta = np.vectorize(lambda x : p['A'] + (p['K']-p['A']) / (1 + 1 * np.exp(-p['B']*(x-p['x0']))))

    # Interpolate the temperature onto the year
    td = pd.date_range(start=datetime.datetime(2022, 1, 1), end=datetime.datetime(2023, 1, 1), freq='M').days_in_month
    ts = np.concatenate((td[-1:], td, td[:1])) / 2
    dt = ts + np.r_[ts[1:], ts[0]]
    tt = np.cumsum(dt)
    f = interp1d(tt  - ts[0] - dt[0], np.concatenate((average_max_temperatur_dk[-1:], average_max_temperatur_dk, average_max_temperatur_dk[:1])), kind='cubic')

    # Apply the temperature model on a monthly basis
    m = 1 + (np.arange(12) - 1 + start_month) % 12
    tm = lambda m : np.arange(td[m-1]) + np.cumsum(td)[m-1] - 31

    beta_per_day = [beta(f(tm(m))) for m in m]

    beta_month_mean = np.array([np.mean(m) for m in beta_per_day])
    beta_month_std  = np.array([np.std(m)  for m in beta_per_day])

    beta_month_std  /= beta_month_mean[np.argmax(m == ref_month)]
    beta_month_mean /= beta_month_mean[np.argmax(m == ref_month)]

    return beta_month_mean, beta_month_std

@st.cache
def vaccined_persons(p_mRNA, d_p_mRNA, p_az, d_p_az, p_jj, d_p_jj, d_vaccine_endorsement) :

    # Load the age data
    N_max = np.loadtxt(os.path.join('data', 'N_max.csv'))
    age_matrix = np.loadtxt(os.path.join('data/', 'age_matrix.csv'))

    # Count total vaccinated (data)
    V_1_mRNA = np.loadtxt(os.path.join('data', 'V_1_mRNA_done.csv'))
    V_1_az   = np.loadtxt(os.path.join('data', 'V_1_az_done.csv'))
    V_1_jj   = np.loadtxt(os.path.join('data', 'V_1_jj_done.csv'))

    # Add total vaccinated (projected)
    V_1_mRNA_p = np.loadtxt(os.path.join('data', 'V_1_mRNA_projected.csv'))
    V_1_az_p   = np.loadtxt(os.path.join('data', 'V_1_az_projected.csv'))
    V_1_jj_p   = np.loadtxt(os.path.join('data', 'V_1_jj_projected.csv'))

    d_V_1_mRNA = V_1_mRNA_p * d_vaccine_endorsement
    d_V_1_az   = V_1_az_p   * d_vaccine_endorsement
    d_V_1_jj   = V_1_jj_p   * d_vaccine_endorsement

    # Add the extra target group
    V_1_mRNA = np.append(V_1_mRNA + V_1_mRNA_p, 0)
    V_1_az   = np.append(V_1_az   + V_1_az_p,   0)
    V_1_jj   = np.append(V_1_jj   + V_1_jj_p,   0)

    d_V_1_mRNA = np.append(d_V_1_mRNA, 0)
    d_V_1_az   = np.append(d_V_1_az,   0)
    d_V_1_jj   = np.append(d_V_1_jj,   0)

    # Project målgrupper onto age groups
    V_1_mRNA_alder = np.dot(V_1_mRNA, age_matrix)
    V_1_az_alder   = np.dot(V_1_az,   age_matrix)
    V_1_jj_alder   = np.dot(V_1_jj,   age_matrix)

    d_V_1_mRNA_alder = np.dot(d_V_1_mRNA, age_matrix)
    d_V_1_az_alder   = np.dot(d_V_1_az,   age_matrix)
    d_V_1_jj_alder   = np.dot(d_V_1_jj,   age_matrix)

    # Project målgrupper counts to age
    N_max_alder = np.dot(N_max, age_matrix)


    # Count the number of effective vaccinated
    V_eff_alder, d_V_eff_alder = error_propergation_addition(
                                        error_propergation_multiplication((V_1_mRNA_alder, d_V_1_mRNA_alder), (p_mRNA, d_p_mRNA)),
                                        error_propergation_multiplication((V_1_az_alder,   d_V_1_az_alder),   (p_az,   d_p_az)),
                                        error_propergation_multiplication((V_1_jj_alder,   d_V_1_jj_alder),   (p_jj,   d_p_jj)))

    V_ineff_alder, d_V_ineff_alder = error_propergation_addition(
                                        error_propergation_multiplication((V_1_mRNA_alder, d_V_1_mRNA_alder), (1-p_mRNA, d_p_mRNA)),
                                        error_propergation_multiplication((V_1_az_alder,   d_V_1_az_alder),   (1-p_az,   d_p_az)),
                                        error_propergation_multiplication((V_1_jj_alder,   d_V_1_jj_alder),   (1-p_jj,   d_p_jj)))


    # Compute the fraction of each age group that is in the vacicnated states
    S_vacc_eff   = V_eff_alder   / N_max_alder
    d_S_vacc_eff = d_V_eff_alder / N_max_alder

    S_vacc_ineff   = V_ineff_alder   / N_max_alder
    d_S_vacc_ineff = d_V_ineff_alder / N_max_alder

    return S_vacc_eff, d_S_vacc_eff, S_vacc_ineff, d_S_vacc_ineff


@st.cache
def prepare_activty_variables() :

    beta_basic  = np.loadtxt(open(os.path.join('data', 'betabasis.txt'),  'rb'), delimiter=' ', skiprows=0)
    population  = np.loadtxt(open(os.path.join('data', 'befolkning.txt'), 'rb'), delimiter=',', skiprows=0)

    age_demograpic = np.sum(population, axis=0)

    n_age_groups = 9

    gamma_EI =  1 / 3.5 #LAEC # 1/Latent period
    rec_I    =  1 / 4.3       # 1/days before recovery or hospital

    Gsmall = np.matrix([[-2*gamma_EI, 2*gamma_EI, 0,          0],
                        [0,          -2*gamma_EI, 2*gamma_EI, 0],
                        [0,          0,          -2*rec_I,    2*rec_I],
                        [0,          0,           0,         -2*rec_I]])


    G  = np.kron(Gsmall,                            np.identity(n_age_groups))
    Pi = np.kron(np.matrix([[1, 0, 0, 0]]),         np.identity(n_age_groups))
    B  = np.kron(np.matrix([[0], [0], [1], [1]]),   np.identity(n_age_groups))

    return G, B, Pi, beta_basic, age_demograpic, n_age_groups


@st.cache(suppress_st_warning=True)
def compute_activity(Nit) :

    # Set random seed
    np.random.seed(0)

    init_grow    = 1.2
    d_init_grow  = 0.01

    G, B, Pi, beta_basic, age_demograpic, _ = prepare_activty_variables()
    age_demograpic_diag = np.diag(age_demograpic / age_demograpic.sum())

    r0 = lambda act, growth : np.abs(np.exp(max(np.real(np.linalg.eigvals(generator_matrix(G, B, Pi, beta_basic*act, age_demograpic_diag))))) - growth)

    activities = np.zeros(Nit)

    for i in tqdm(range(Nit)) :

        growth_use = np.random.normal(loc=init_grow, scale=d_init_grow, size=None)

        activities[i] = minimize_scalar(r0, bounds=(0, 1), args=growth_use, method='bounded').x # Find activity

    return activities


def generator_matrix(G, B, Pi, total_activity, S_matrix_age) :
    return G + B.dot(total_activity).dot(S_matrix_age).dot(Pi)


def eigenvector(S_vec, beta_multiplier, normalize=True):

    G, B, Pi, beta_basic, age_demograpic, n_age_groups = prepare_activty_variables()

    # Construct the generator matrix
    g_M = generator_matrix(G, B, Pi, beta_basic * compute_activity(1_000).mean() * beta_multiplier, np.diag(age_demograpic * S_vec / age_demograpic.sum()))

    EigSys = eig(g_M, left=True, right=False)

    eigvecState = np.real(EigSys[1][:, np.argmax(np.real(EigSys[0]))]) # Eigenvector for all states

    # Sum up the 4 states into one measure of infection. (So sum of E1, E2, I1, and I2.)
    eigvec  =  eigvecState[0:n_age_groups]
    eigvec  += eigvecState[  n_age_groups:2*n_age_groups]
    eigvec  += eigvecState[2*n_age_groups:3*n_age_groups]
    eigvec  += eigvecState[3*n_age_groups:4*n_age_groups]

    if normalize :
        eigvec = eigvec / eigvec.sum()
    else :
        v = np.dot(eigvecState, g_M)
        eigvec  =  v[0:n_age_groups]
        eigvec  += v[  n_age_groups:2*n_age_groups]
        eigvec  += v[2*n_age_groups:3*n_age_groups]
        eigvec  += v[3*n_age_groups:4*n_age_groups]

    return eigvec


def R_t_activity(S_vec, d_S_vec) :

    Nit = 1_000 # How many samples do we make?

    G, B, Pi, beta_basic, age_demograpic, _ = prepare_activty_variables()

    contact_vector = np.empty(Nit)

    for it, activity in enumerate(compute_activity(Nit)):

        tmp_S_vec = age_demograpic / age_demograpic.sum() * np.clip(np.random.normal(loc=S_vec, scale=d_S_vec), 0, 1)

        # Next generation matrix A without the susceptible vector S (From matrix formalism of SEIR)
        AnoS = - Pi.dot( np.linalg.inv(G)).dot(B).dot(beta_basic) * activity

        contact_vector[it] = max(np.real(np.linalg.eigvals(AnoS.dot(np.diag(tmp_S_vec)))))


    return np.mean(contact_vector), np.std(contact_vector) / np.sqrt(Nit)





class tqdm:
    def __init__(self, iterable, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)

        self.prog_bar.empty()