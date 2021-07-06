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

from helpers import error_propergation_multiplication, temperature_model, vaccined_persons, R_t_activity, prepare_activty_variables, eigenvector, minimize_sero, pop_avg, load_risk

plt.style.use('ggplot')

########     ###    ########     ###    ##     ## ######## ######## ######## ########   ######
##     ##   ## ##   ##     ##   ## ##   ###   ### ##          ##    ##       ##     ## ##    ##
##     ##  ##   ##  ##     ##  ##   ##  #### #### ##          ##    ##       ##     ## ##
########  ##     ## ########  ##     ## ## ### ## ######      ##    ######   ########   ######
##        ######### ##   ##   ######### ##     ## ##          ##    ##       ##   ##         ##
##        ##     ## ##    ##  ##     ## ##     ## ##          ##    ##       ##    ##  ##    ##
##        ##     ## ##     ## ##     ## ##     ## ########    ##    ######## ##     ##  ######

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

transmission_reduction_vaccinated   = (1 - 0.49) # among vaccinated persons # Pfizer - https://khub.net/documents/135939561/390853656/Impact+of+vaccination+on+household+transmission+of+SARS-COV-2+in+England.pdf/35bf4bb1-6ade-d3eb-a39e-9c9b25a8122a
d_transmission_reduction_vaccinated = np.abs(np.array([0.44, 0.56]) - 0.49).mean() / CIs_to_std  # uncertainty on this number

transmission_reduction_children   = 0.5
d_transmission_reduction_children = 0

d_behavior = 0.1

risk_reduction = 0.75
alpha_risk     = 1.4
delta_risk     = 2.0


########     ###     ######  ##     ## ########   #######     ###    ########  ########
##     ##   ## ##   ##    ## ##     ## ##     ## ##     ##   ## ##   ##     ## ##     ##
##     ##  ##   ##  ##       ##     ## ##     ## ##     ##  ##   ##  ##     ## ##     ##
##     ## ##     ##  ######  ######### ########  ##     ## ##     ## ########  ##     ##
##     ## #########       ## ##     ## ##     ## ##     ## ######### ##   ##   ##     ##
##     ## ##     ## ##    ## ##     ## ##     ## ##     ## ##     ## ##    ##  ##     ##
########  ##     ##  ######  ##     ## ########   #######  ##     ## ##     ## ########


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


include_vaccines = st.sidebar.checkbox('Vaccines',              value=True)
if include_vaccines :
    include_reduced_transmission_among_vaccinated           = st.sidebar.checkbox('Reduced beta (vaccinated)',  value=True, help=f'Reduction of transmission ({transmission_reduction_vaccinated:.1f}) among people with vaccine failure')
    vaccine_endorsement_12_15 = st.sidebar.slider('Endorsement 12-15 y.olds', 0.0, 1.0, 0.9,     help='Proportion of 12-15 year olds who accept vaccination') / 0.9
else :
    include_reduced_transmission_among_vaccinated           = False
    vaccine_endorsement_12_15 = 0

include_season   = st.sidebar.checkbox('Season',                value=True)
include_reduced_transmission_among_children = st.sidebar.checkbox('Reduced beta (children)', value=False, help=f'Reduction of transmission ({transmission_reduction_children:.1f}) among children. NB: There is currenly some double counting of this effect when enabled')
include_behavior = st.sidebar.checkbox('Behavior change',       value=True, help=f'Added uncertainty of +- {d_behavior}')

tracing          = st.sidebar.slider('Tracing reduction', 0.0, 0.25, 0.1, help='Uncertainty is set the same as the value')


if option == 'March 2020' :
    sero_model_order = st.sidebar.slider('Seroprevalence model order', 0, 5, 5, step=1, help='Number of iterative steps taken along eigenvector direction')
    ref_month = 3   # March (Starting Rt computed in the march)
elif option == 'Fall 2020' :
    sero_model_order = 0
    ref_month = 10   # October (Starting Rt computed in the "fall")

plot_only_seroprevalence = st.sidebar.checkbox('Show only seroprevalence', value=False)



##        #######     ###    ########     ########     ###    ########    ###
##       ##     ##   ## ##   ##     ##    ##     ##   ## ##      ##      ## ##
##       ##     ##  ##   ##  ##     ##    ##     ##  ##   ##     ##     ##   ##
##       ##     ## ##     ## ##     ##    ##     ## ##     ##    ##    ##     ##
##       ##     ## ######### ##     ##    ##     ## #########    ##    #########
##       ##     ## ##     ## ##     ##    ##     ## ##     ##    ##    ##     ##
########  #######  ##     ## ########     ########  ##     ##    ##    ##     ##


# Load data for the app
_, _, _, _, age_demograpic, n_age_groups = prepare_activty_variables()

# Load data for the vaccine rollout
if include_vaccines :
    S_vacc_eff, d_S_vacc_eff, S_vacc_ineff, d_S_vacc_ineff = vaccined_persons(p_mRNA, d_p_mRNA, p_az, d_p_az, p_jj, d_p_jj, d_vaccine_endorsement, vaccine_endorsement_12_15)
else :
    S_vacc_eff, d_S_vacc_eff, S_vacc_ineff, d_S_vacc_ineff = (0.0, 0.0, 0.0, 0.0)


# Get temperature componnent
R_temp, d_R_temp = temperature_model(start_month, ref_month)


# Get the distribution of sero prevalence (based on week 16)
# Starting point is 10 % infected
sero_current = 0.1
sero_0 = minimize_sero(n_age_groups, sero_current)



 ######  ##     ## ########  ########  ######## ##    ## ########    #### ##     ## ##     ## ##     ## ##    ## #### ######## ##    ##
##    ## ##     ## ##     ## ##     ## ##       ###   ##    ##        ##  ###   ### ###   ### ##     ## ###   ##  ##     ##     ##  ##
##       ##     ## ##     ## ##     ## ##       ####  ##    ##        ##  #### #### #### #### ##     ## ####  ##  ##     ##      ####
##       ##     ## ########  ########  ######   ## ## ##    ##        ##  ## ### ## ## ### ## ##     ## ## ## ##  ##     ##       ##
##       ##     ## ##   ##   ##   ##   ##       ##  ####    ##        ##  ##     ## ##     ## ##     ## ##  ####  ##     ##       ##
##    ## ##     ## ##    ##  ##    ##  ##       ##   ###    ##        ##  ##     ## ##     ## ##     ## ##   ###  ##     ##       ##
 ######   #######  ##     ## ##     ## ######## ##    ##    ##       #### ##     ## ##     ##  #######  ##    ## ####    ##       ##


# Divide the population by immunity and vaccination status
with_immunity          = S_vacc_eff  + (1 - S_vacc_eff) * sero_0   # Fraction of people with immunity
d_with_immunity        = (1  - sero_0) * d_S_vacc_eff              # Uncertainty

no_immunity            = 1 - with_immunity                         # Fraction of people without immunity
d_no_immunity          = d_with_immunity                           # Uncertainty

vaccinated_no_immunity   = S_vacc_ineff * (1 - sero_0)             # Fraction of vaccinated people without immunity
d_vaccinated_no_immunity = d_S_vacc_ineff



# People who are vulnerable and vaccinated have reduced infection spread which modifies the effective beta
if include_reduced_transmission_among_vaccinated :
    beta_vaccine_failure   = (no_immunity + vaccinated_no_immunity * transmission_reduction_vaccinated) / (no_immunity + vaccinated_no_immunity)
    d_beta_vaccine_failure = np.sqrt(np.power( ((1                      - beta_vaccine_failure) / (beta_vaccine_failure * (no_immunity + vaccinated_no_immunity)) ) * d_no_immunity, 2) +
                                     np.power( ((transmission_reduction_vaccinated - beta_vaccine_failure) / (beta_vaccine_failure * (no_immunity + vaccinated_no_immunity)) ) * d_vaccinated_no_immunity, 2) +
                                     np.power( ( vaccinated_no_immunity / (no_immunity + vaccinated_no_immunity) ) * d_transmission_reduction_vaccinated, 2))

else :
    beta_vaccine_failure   = np.ones(n_age_groups)
    d_beta_vaccine_failure = np.zeros(n_age_groups)


# Children can have reduced infection spread which modifies the effective beta
if include_reduced_transmission_among_children :
    # age profile of children who have reduced transmission
    #                                   0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79,   80+
    children_age_profile   = np.array([ 1.0,  0.10,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0])   # 0.1 = 1 age groups (10 yrs old)
    children_age_profile  += np.array([ 0.0,  0.05*(1-vaccine_endorsement_12_15), 0,0,0,0,0,0,0    ])   # 0.05 = 0.5 age groups (half of 11 yrs old - some will be offered vaccine)
    d_children_age_profile = np.array([ 0.0,  0.05,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]) * d_vaccine_endorsement

    beta_children   = (1-children_age_profile) + children_age_profile * transmission_reduction_children
    d_beta_children = np.sqrt(np.power( (1 - transmission_reduction_children) * d_children_age_profile, 2) +
                              np.power(  children_age_profile * d_transmission_reduction_children, 2))

else :
    beta_children   = np.ones(n_age_groups)
    d_beta_children = np.zeros(n_age_groups)


########  ########   #######        ## ########  ######  ########     ######  ######## ########   #######
##     ## ##     ## ##     ##       ## ##       ##    ##    ##       ##    ## ##       ##     ## ##     ##
##     ## ##     ## ##     ##       ## ##       ##          ##       ##       ##       ##     ## ##     ##
########  ########  ##     ##       ## ######   ##          ##        ######  ######   ########  ##     ##
##        ##   ##   ##     ## ##    ## ##       ##          ##             ## ##       ##   ##   ##     ##
##        ##    ##  ##     ## ##    ## ##       ##    ##    ##       ##    ## ##       ##    ##  ##     ##
##        ##     ##  #######   ######  ########  ######     ##        ######  ######## ##     ##  #######

if sero_prevalence <= sero_current :    # If target prevalence is less than the estimated current value, scale the starting vector
    sero_0 = minimize_sero(n_age_groups, sero_prevalence)


if sero_model_order > 0 and option == 'March 2020' :

    if sero_prevalence > sero_current :

        # Iteratively take sero_model_order steps along the eigenvectors to determine sero prevalence
        tmp_sero = sero_0
        sero_step = (sero_prevalence - sero_current) / sero_model_order

        for i in range(sero_model_order) :

            # Update the temporary S protected vecor
            tmp_S_protected = S_vacc_eff + (1 - S_vacc_eff) * sero_0
            tmp_S_protected += np.clip(tmp_sero - sero_0, 0, 1-tmp_S_protected)

            # Determine the eigen vector and take a step along that direction
            v = eigenvector(1 - tmp_S_protected, rel_beta_alpha * rel_beta_delta * R_temp.max() * beta_vaccine_failure * beta_children)
            fun = lambda k : pop_avg(np.clip(tmp_sero + k * v, 0, 1)) - ((i+1) * sero_step + sero_current)
            try :
                k = scipy.optimize.root_scalar(fun, bracket=(0, 10_000)).root
            except :
                st.error('Seroprevalence too high: cannot compute projection')
                break

            tmp_sero = np.clip(tmp_sero + k * v, 0, 1)


        # Set the sero prevalence to the fitted value
        sero_prevalence = tmp_sero

    else :
        sero_prevalence = sero_0

else :
    fun = lambda k : pop_avg(np.maximum(np.ones(n_age_groups) * k, sero_0)) - sero_prevalence
    k = scipy.optimize.root_scalar(fun, bracket=(0, 10_000)).root
    sero_prevalence = np.maximum(np.ones(n_age_groups) * k, sero_0)



 ######          ##     ## ########  ######
##    ##         ##     ## ##       ##    ##
##               ##     ## ##       ##
 ######          ##     ## ######   ##
      ##          ##   ##  ##       ##
##    ##           ## ##   ##       ##    ##
 ######  #######    ###    ########  ######

# Get the susceptible population
# Starting point
S_protected = np.zeros(n_age_groups)

# People with effective vaccines are protected
S_protected += S_vacc_eff

# People without effective vacines are protected if they had the desease before
S_protected += (1 - S_protected) * sero_0
S_protected += np.clip(sero_prevalence - sero_0, 0, 1-S_protected)

# Determine the vulnerable people and the uncertainty
S_vec   = 1 - S_protected
d_S_vec = d_S_vacc_eff

if np.any(S_vec == 0) :
    st.warning('Seroprevalence too high: unreliable projection')


#### ##    ## ######## ########  ######  ######## ######## ########
 ##  ###   ## ##       ##       ##    ##    ##    ##       ##     ##
 ##  ####  ## ##       ##       ##          ##    ##       ##     ##
 ##  ## ## ## ######   ######   ##          ##    ######   ##     ##
 ##  ##  #### ##       ##       ##          ##    ##       ##     ##
 ##  ##   ### ##       ##       ##    ##    ##    ##       ##     ##
#### ##    ## ##       ########  ######     ##    ######## ########


# Compute the number of infected within each age group
added_seroprevalence     = sero_prevalence - sero_0
n_infected               = age_demograpic * added_seroprevalence
n_infected_vaccinated    = age_demograpic * vaccinated_no_immunity * added_seroprevalence
n_infected_nonvaccinated = n_infected - n_infected_vaccinated



##     ##  #######   ######  ########  #### ########    ###    ##       #### ######## ######## ########
##     ## ##     ## ##    ## ##     ##  ##     ##      ## ##   ##        ##       ##  ##       ##     ##
##     ## ##     ## ##       ##     ##  ##     ##     ##   ##  ##        ##      ##   ##       ##     ##
######### ##     ##  ######  ########   ##     ##    ##     ## ##        ##     ##    ######   ##     ##
##     ## ##     ##       ## ##         ##     ##    ######### ##        ##    ##     ##       ##     ##
##     ## ##     ## ##    ## ##         ##     ##    ##     ## ##        ##   ##      ##       ##     ##
##     ##  #######   ######  ##        ####    ##    ##     ## ######## #### ######## ######## ########


# Compute the number of infected within each age group
risk = load_risk()

n_hospitalized_nonvaccinated = alpha_risk * delta_risk * risk * n_infected_nonvaccinated
n_hospitalized_vaccinated    = alpha_risk * delta_risk * risk * n_infected_vaccinated * (1 - risk_reduction)
n_hospitalized = n_hospitalized_nonvaccinated + n_hospitalized_vaccinated




########          ########
##     ##            ##
##     ##            ##
########             ##
##   ##              ##
##    ##             ##
##     ## #######    ##

if option == 'Fall 2020' :
    # Starting point
    Rt   = 1.13
    d_Rt = 0.26


    Rt, d_Rt = error_propergation_multiplication((Rt, d_Rt),
                                                (pop_avg(S_vec), pop_avg(d_S_vec)),                                 # Add immunity
                                                (pop_avg(beta_vaccine_failure), pop_avg(d_beta_vaccine_failure)),   # Add reduced transmission from vaccine failure
                                                (pop_avg(beta_children), pop_avg(d_beta_children)))                 # Add the reduced transmission among children

elif option == 'March 2020' :

    # Gather the activity modifiers
    activity_modifier   = beta_vaccine_failure * beta_children
    d_activity_modifier = np.sqrt(np.power(beta_children * d_beta_vaccine_failure, 2) +
                                  np.power(beta_vaccine_failure * d_beta_children, 2))


    # Get activity from model
    Rt, d_Rt  = R_t_activity(S_vec, d_S_vec, activity_modifier, d_activity_modifier)

# Compute the contact number
Rt, d_Rt = error_propergation_multiplication(
        (Rt,             d_Rt,                      True),
        (rel_beta_alpha, d_rel_beta_alpha,          include_alpha),         # Beta variant
        (rel_beta_delta, d_rel_beta_delta,          include_delta),         # Delta variant
        (1-tracing,      tracing,                   True),                  # Contact tracing
        (1.0,            d_behavior,                include_behavior),      # Changes in behavior
        (R_temp,         d_R_temp,                  include_season)         # Temperature
        )





######## ####  ######   ##     ## ########  ########  ######
##        ##  ##    ##  ##     ## ##     ## ##       ##    ##
##        ##  ##        ##     ## ##     ## ##       ##
######    ##  ##   #### ##     ## ########  ######    ######
##        ##  ##    ##  ##     ## ##   ##   ##             ##
##        ##  ##    ##  ##     ## ##    ##  ##       ##    ##
##       ####  ######    #######  ##     ## ########  ######



ymax = max(2, np.ceil(np.max(Rt + d_Rt)))

# Build the main plot
t = pd.date_range(start=datetime.datetime(2021, start_month-1, 1), periods=12, freq='M') + pd.Timedelta(days=1)

fig = plt.figure(figsize=(6.4, 4))
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

st.pyplot(fig)


fig = plt.figure(figsize=(6.4, 2))
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

st.pyplot(fig)




fig = plt.figure(figsize=(6.4, 2))
ax = plt.gca()
ax.bar(np.arange(n_age_groups), n_infected,             color=plt.cm.tab10(3), label=f'Total: {int(n_infected.sum()):,}'.replace(',', '.'))
ax.bar(np.arange(n_age_groups), n_infected_vaccinated,  color=plt.cm.tab10(4), label=f'Among vaccinated: {int(n_infected_vaccinated.sum()):,}'.replace(',', '.'))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=2, fontsize=8)
ax.set_xticks(np.arange(n_age_groups))
labels = [f'{10*i}-{10*(i+1)-1}' for i in np.arange(n_age_groups)]
labels[-1] = f'{10*n_age_groups}+'
ax.set_xticklabels(labels)
ax.set_xlabel('age group')
ax.set_ylabel('new infected')
ax.set_ylim(0, 400_000)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

st.pyplot(fig)


fig = plt.figure(figsize=(6.4, 2))
ax = plt.gca()
ax.bar(np.arange(n_age_groups), n_hospitalized,            color=plt.cm.tab10(3), label=f'Total: {int(n_hospitalized.sum()):,}'.replace(',', '.'))
ax.bar(np.arange(n_age_groups), n_hospitalized_vaccinated, color=plt.cm.tab10(4), label=f'Among vaccinated: {int(n_hospitalized_vaccinated.sum()):,}'.replace(',', '.'))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), shadow=False, ncol=2, fontsize=8)
ax.set_xticks(np.arange(n_age_groups))
labels = [f'{10*i}-{10*(i+1)-1}' for i in np.arange(n_age_groups)]
labels[-1] = f'{10*n_age_groups}+'
ax.set_xticklabels(labels)
ax.set_xlabel('age group')
ax.set_ylabel('new hospitalizations')
ax.set_ylim(0, 10_000)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

st.pyplot(fig)
