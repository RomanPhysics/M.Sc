#Roman Sultanov
#Imports
from AmpLambda import *
#from AmpXi import *
Px, Py, Pz = 0.0, 0.0, 0.95
HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUp0r, HUp0i, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i \
    =0.29, 0.04, -0.16, 1.5, -6.8, 3.1, -13, 4.5, 1.0, 0.0, 1.19, -1.03, -3.1, -3.3, -0.7, -4.2

@njit
def generate(num_samples):
    samples = []
    while len(samples) < num_samples:
        m2pK = np.random.uniform((mp + mK) ** 2, (mΛ - mπ) ** 2)
        m2Kπ = np.random.uniform((mK + mπ) ** 2, (mΛ - mp) ** 2)
        
        Es2 = (m2pK - mp ** 2 + mK ** 2) / (2 * np.sqrt(m2pK))
        Es3 = (mΛ ** 2 - m2pK - mπ ** 2) / (2 * np.sqrt(m2pK))
        sqrt_term1 = np.sqrt(Es2 ** 2 - mK ** 2)
        sqrt_term2 = np.sqrt(Es3 ** 2 - mπ ** 2)
        sq_term = (Es2 + Es3) ** 2
        if m2pK <= (mp + mK) ** 2 or m2pK >= (mΛ - mπ) ** 2 or m2Kπ <= sq_term - (sqrt_term1 + sqrt_term2) ** 2 or m2Kπ >= sq_term - (sqrt_term1 - sqrt_term2) ** 2:
            continue

        cos_theta_p = np.random.uniform(-1.0, 1.0)
        phi_p = np.random.uniform(-np.pi, np.pi)
        chi = np.random.uniform(-np.pi, np.pi)

        pdf = PDF(m2pK, m2Kπ, cos_theta_p, phi_p, chi, Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUp0r, HUp0i, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i)
        
        ymax=0.082
        y = np.random.uniform(0.0, ymax)
        if pdf >0.082:
            print(f'Hyperbox does not enclose PDF. Current ceiling {ymax}, PDF exceeded with {pdf}')
            print([m2pK, m2Kπ, cos_theta_p, phi_p, chi])
        if y < pdf:
            samples.append([m2pK, m2Kπ, cos_theta_p, phi_p, chi])
    return np.array(samples)

import csv
def work():
    samples = generate(1000000)
    with open('MCData.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(samples)
work()