import pandas as pd
import numpy as np
import os

rollout_date   = '2021_06_14'
age_group_date = '2021_06_08'

lastweek = 22   # Last week of vaccine data


# Vaccine rollout

# Load calendar
filename_rollout = f'VaccinationsKalender_{rollout_date}.xlsx'
tmp = pd.read_excel(filename_rollout, sheet_name = 1)
N_max = np.round(tmp['Antal (justeret)'].to_numpy())

np.savetxt(os.path.join('data', 'N_max.csv'), N_max)

K_1_mRNA  = pd.read_excel(filename_rollout, sheet_name = 'Kalender_1_Stik_1_mRNA', index_col = 0)
K_1_az    = pd.read_excel(filename_rollout, sheet_name = 'Kalender_1_Stik_1_AZ', index_col = 0)
K_1_jj    = pd.read_excel(filename_rollout, sheet_name = 'Kalender_1_Stik_1_JJ', index_col = 0)

# Load the age data
age = pd.read_excel(f'Vaccination_maalgr_{age_group_date}.xlsx', index_col=0)
age[age.isna()] = 0
age = age.to_numpy().astype(float)
age_matrix = (age / np.repeat(np.sum(age, axis=1).reshape((17, 1)), 9, axis=1))

np.savetxt('age_matrix.csv', age_matrix)


# Count total vaccinated (data)
V_1_mRNA = K_1_mRNA[[53] + [i for i in range(1, lastweek)]].sum(axis=1)
V_1_az   = K_1_az[  [53] + [i for i in range(1, lastweek)]].sum(axis=1)
V_1_jj   = K_1_jj[  [53] + [i for i in range(1, lastweek)]].sum(axis=1)


np.savetxt(os.path.join('data', 'V_1_mRNA_done.csv'), V_1_mRNA)
np.savetxt(os.path.join('data', 'V_1_az_done.csv'),   V_1_az)
np.savetxt(os.path.join('data', 'V_1_jj_done.csv'),   V_1_jj)


# Add total vaccinated (projected)
V_1_mRNA_p = K_1_mRNA[[i for i in range(lastweek+1, K_1_mRNA.columns[1:].max()+1)]].sum(axis=1)
V_1_az_p   =   K_1_az[[i for i in range(lastweek+1, K_1_mRNA.columns[1:].max()+1)]].sum(axis=1)
V_1_jj_p   =   K_1_jj[[i for i in range(lastweek+1, K_1_mRNA.columns[1:].max()+1)]].sum(axis=1)

np.savetxt(os.path.join('data', 'V_1_mRNA_projected.csv'), V_1_mRNA_p)
np.savetxt(os.path.join('data', 'V_1_az_projected.csv'),   V_1_az_p)
np.savetxt(os.path.join('data', 'V_1_jj_projected.csv'),   V_1_jj_p)
