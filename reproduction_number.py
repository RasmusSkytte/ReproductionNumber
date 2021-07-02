from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import streamlit as st

import scipy
from scipy.interpolate import interp1d

import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

import datetime

from helpers import error_propergation_multiplication, temperature_model, vaccined_persons, R_t_activity, prepare_activty_variables, eigenvector

plt.style.use('ggplot')


# Parameters for the visualization
start_month = 9     # October (Start of the visualization)

CIs_to_std = scipy.special.erfcinv(1-0.95) * np.sqrt(2)

rel_beta_alpha   = 1.55
d_rel_beta_alpha = 0.05

#rel_beta_delta   = 1.64 # Technical briefing 15
#d_rel_beta_delta = np.abs(np.array([1.26, 2.13]) - 1.64).mean() / CIs_to_std  # 95% CI to std
d_rel_beta_delta = 0.2

d_vaccine_endorsement = 0.05    # Uncertainty on number of people who will be vaccinated

p_mRNA   = 0.9  # mRNA efficacy
d_p_mRNA = 0.05 # Uncertainty on efficay

p_az   = 0.6    # az efficacy
d_p_az = 0.05   # Uncertainty on efficay

p_jj   = 0.6    # jj efficacy
d_p_jj = 0.05   # Uncertainty on efficay

transmission_reduction   = (1 - 0.49) # among vaccinated persons # Pfizer - https://khub.net/documents/135939561/390853656/Impact+of+vaccination+on+household+transmission+of+SARS-COV-2+in+England.pdf/35bf4bb1-6ade-d3eb-a39e-9c9b25a8122a
d_transmission_reduction = np.abs(np.array([0.44, 0.56]) - 0.49).mean() / CIs_to_std  # uncertainty on this number


d_behavior = 0.1


# Build interface
st.title('Reproduction number')


# Build the sidebar
option = st.sidebar.selectbox('Reproduction number reference', ('Fall 2020', 'March 2020'))
sero_prevalence  = st.sidebar.slider('Seroprevalence through infection (%)', 0, 100, 10) / 100

variants = st.sidebar.radio('Variants to include', ('Only WT', '+ alpha', '+ alpha + delta'), index = 2)

include_alpha = False
include_delta = False
if variants == '+ alpha' :
    include_alpha = True
elif variants == '+ alpha + delta' :
    include_alpha = True
    include_delta = True

if include_delta :
    rel_beta_delta = st.sidebar.slider('Rel. delta rate', 1.5, 2.2, 1.9, help='Growth rate relative to alpha variant')
else :
    rel_beta_delta = 1.0


tracing          = st.sidebar.slider('Tracing reduction', 0.0, 0.25, 0.1, help='Uncertainty is set the same as the value')
include_vaccines = st.sidebar.checkbox('Include vaccines',          value=True)
include_season   = st.sidebar.checkbox('Include season',            value=True)
include_behavior = st.sidebar.checkbox('Include behavior change',   value=True, help=f'Added uncertainty of +- {d_behavior}')
if option == 'March 2020' :
    sero_model_order = st.sidebar.slider('Seroprevalence model order', 0, 5, 5, step=1, help='Number of iterative steps taken along eigenvector direction')
    ref_month = 3   # March (Starting Rt computed in the march)
elif option == 'Fall 2020' :
    sero_model_order = 0
    ref_month = 10   # October (Starting Rt computed in the "fall")

plot_only_seroprevalence = st.sidebar.checkbox('Show only seroprevalence', value=False)

# Load data for the app
_, _, _, _, age_demograpic, n_age_groups = prepare_activty_variables()
pop_avg = lambda x : np.sum(x * age_demograpic) / age_demograpic.sum()


# Load data for the vaccine rollout
if include_vaccines :
    S_vacc_eff, d_S_vacc_eff, S_vacc_ineff, d_S_vacc_ineff = vaccined_persons(p_mRNA, d_p_mRNA, p_az, d_p_az, p_jj, d_p_jj, d_vaccine_endorsement)
else :
    S_vacc_eff, d_S_vacc_eff, S_vacc_ineff, d_S_vacc_ineff = (0.0, 0.0, 0.0, 0.0)


# Get temperature componnent
R_temp, d_R_temp = temperature_model(start_month, ref_month)



# Get the distribution of sero prevalence (based on week 16)
x = [(29-17)/2+17, (49-30)/2+30, (69-50)/2+50]
y = [0.096, 0.063, 0.050]

f = interp1d(x, y, kind='nearest', fill_value='extrapolate')
xx = np.arange(5, n_age_groups*10, 10)
sero_0 = f(xx)
minimize_sero = lambda level : sero_0 * scipy.optimize.root_scalar(lambda k : pop_avg(sero_0 * k) - level, bracket=(0, 10)).root


# Get the susceptible population
# Starting point
S_protected = np.zeros(n_age_groups)

# People with effective vaccines are protected
S_protected += S_vacc_eff

# Starting point is 10 % infected
sero_current = 0.1
sero_0 = minimize_sero(sero_current)


if sero_prevalence <= sero_current :    # If target prevalence is less than the estimated current value, scale the starting vector
    sero_0 = minimize_sero(sero_prevalence)


if sero_model_order > 0 and option == 'March 2020' :

    if sero_prevalence > sero_current :

        # Iteratively take sero_model_order steps along the eigenvectors to determine sero prevalence
        tmp_sero = sero_0
        sero_step = (sero_prevalence - sero_current) / sero_model_order

        for i in range(sero_model_order) :

            # Update the temporary S protected vecor
            tmp_S_protected = S_protected + (1 - S_protected) * sero_0 + np.clip(tmp_sero - sero_0, 0, 1) + transmission_reduction * S_vacc_ineff * (1 - tmp_sero)

            # Determine the eigen vector and take a step along that direction
            v = eigenvector(1 - tmp_S_protected, rel_beta_alpha * rel_beta_delta * R_temp.max())
            fun = lambda k : pop_avg(np.clip(tmp_sero + k * v, 0, 1)) - ((i+1) * sero_step + sero_current)
            k = scipy.optimize.root_scalar(fun, bracket=(0, 10_000)).root
            tmp_sero = np.clip(tmp_sero + k * v, 0, 1)


        # Set the sero prevalence to the fitted value
        sero_prevalence = tmp_sero

    else :
        sero_prevalence = sero_0

else :
    fun = lambda k : pop_avg(np.maximum(np.ones(n_age_groups) * k, sero_0)) - sero_prevalence
    k = scipy.optimize.root_scalar(fun, bracket=(0, 10_000)).root
    sero_prevalence = np.maximum(np.ones(n_age_groups) * k, sero_0)

print('pop_avg(sero_prevalence)')
print(pop_avg(sero_prevalence))

# People without effective vacines are protected if they had the desease before
S_protected += (1 - S_protected) * sero_0 + np.clip(sero_prevalence - sero_0, 0, 1)

# People who are vulnerable and vaccinated have reduced infection spread
S_protected += transmission_reduction * S_vacc_ineff * (1 - sero_prevalence)


S_vec   = 1 - S_protected
d_S_vec = np.sqrt(np.power(d_S_vacc_eff, 2.0)
                + np.power(transmission_reduction * d_S_vacc_ineff, 2.0)
                + np.power(S_vacc_ineff * d_transmission_reduction, 2.0))



if option == 'Fall 2020' :
    # Starting point
    Rt   = 1.13
    d_Rt = 0.26

    # Add immunity
    Rt, d_Rt = error_propergation_multiplication((Rt, d_Rt), (pop_avg(S_vec), pop_avg(d_S_vec)))


elif option == 'March 2020' :

    # Get activity from model
    Rt, d_Rt  = R_t_activity(S_vec, d_S_vec)

# Compute the contact number
Rt, d_Rt = error_propergation_multiplication(
        (Rt,             d_Rt,                      True),
        (rel_beta_alpha, d_rel_beta_alpha,          include_alpha),         # Beta variant
        (rel_beta_delta, d_rel_beta_delta,          include_delta),         # Delta variant
        (1-tracing,      tracing,                   True),                  # Contact tracing
        (1.0,            d_behavior,                include_behavior),      # Changes in behavior
        (R_temp,         d_R_temp,                  include_season)         # Temperature
        )


ymax = max(2, np.ceil(np.max(Rt + d_Rt)))

# Build the main plot
t = pd.date_range(start=datetime.datetime(2021, start_month-1, 1), periods=12, freq='M') + pd.Timedelta(days=1)

fig1 = plt.figure(figsize=(6.4, 4))
ax = plt.gca()
ax.errorbar(t, Rt, yerr=d_Rt, fmt='s', color=plt.cm.tab10(0))
ax.set_xlabel('2021 / 2022')
ax.set_ylabel('$R_t$')
ax.set_ylim(0, ymax)
ax.set_yticks(np.arange(ymax+1))

xlim = ax.get_xlim()
ax.set_xlim(xlim)
ax.plot(xlim, np.ones(2), 'k--')


ax.add_patch(mpl.patches.Rectangle((xlim[0], 0  ), xlim[-1]-xlim[0], 0.8,  zorder=1, color=plt.cm.tab10(2), alpha = 0.3))
ax.add_patch(mpl.patches.Rectangle((xlim[0], 0.8), xlim[-1]-xlim[0], 0.2,  zorder=1, color=plt.cm.tab10(8), alpha = 0.2))
ax.add_patch(mpl.patches.Rectangle((xlim[0], 1.0), xlim[-1]-xlim[0], ymax, zorder=1, color=plt.cm.tab10(3), alpha = 0.3))

months     = mdates.MonthLocator(interval=1)
months_fmt = mdates.DateFormatter('%b')

ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

st.pyplot(fig1)


fig2 = plt.figure(figsize=(6.4, 2))
ax = plt.gca()
if plot_only_seroprevalence :
    ax.bar(np.arange(n_age_groups), sero_prevalence * 100, color=plt.cm.tab10(3), label='Projected seroprevalence')
    ax.bar(np.arange(n_age_groups), sero_0 * 100,          color=plt.cm.tab10(1), label='Current seroprevalence')
    ax.set_ylabel('seroprevalence (%)')
    ncol = 2
else :
    ax.bar(np.arange(n_age_groups), (S_vacc_eff + (1-S_vacc_eff)*sero_0 + sero_prevalence - sero_0) * 100, color=plt.cm.tab10(3), label='Projected seroprevalence')
    ax.bar(np.arange(n_age_groups), (S_vacc_eff + (1-S_vacc_eff)*sero_0) * 100,                            color=plt.cm.tab10(1), label='Current seroprevalence')
    ax.bar(np.arange(n_age_groups),  S_vacc_eff * 100,                                                     color=plt.cm.tab10(2), label='Vaccines')
    ax.set_ylabel('immunity / seroprev. (%)')
    ncol = 3

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=ncol, fontsize=8)


ax.set_xticks(np.arange(n_age_groups))
labels = [f'{10*i}-{10*(i+1)-1}' for i in np.arange(n_age_groups)]
labels[-1] = f'{10*n_age_groups}+'
ax.set_xticklabels(labels)
ax.set_xlabel('age group')
ax.set_ylim(0, 100)

st.pyplot(fig2)



fig3 = plt.figure(figsize=(6.4, 2))
ax = plt.gca()
ax.bar(np.arange(n_age_groups), age_demograpic * (sero_prevalence - sero_0), color=plt.cm.tab10(3))
ax.set_xticks(np.arange(n_age_groups))
labels = [f'{10*i}-{10*(i+1)-1}' for i in np.arange(n_age_groups)]
labels[-1] = f'{10*n_age_groups}+'
ax.set_xticklabels(labels)
ax.set_xlabel('age group')
ax.set_ylabel('new infected')
ax.set_ylim(0, 500_000)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

st.pyplot(fig3)
