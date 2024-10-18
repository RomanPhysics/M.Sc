#Roman Sultanov
#Import AMPL or AMPX
from AMPL import *
#from AMPX import *

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

        cos_theta_p = np.random.uniform(-1, 1)
        phi_p = np.random.uniform(-np.pi, np.pi)
        chi = np.random.uniform(-np.pi, np.pi)

        pdf = DecayrateDist(m2pK, m2Kπ, cos_theta_p, phi_p, chi, Px=0.0, Py=0.0, Pz=0.95, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2)
        
        y = np.random.uniform(0, 0.082)
        if pdf >0.082:
            print(pdf)
            print([m2pK, m2Kπ, cos_theta_p, phi_p, chi])
        if y < pdf:
            samples.append([m2pK, m2Kπ, cos_theta_p, phi_p, chi])
    return np.array(samples)

import csv
def work():
    samples = generate(10000000)
    with open(r'C:\Users\Roman\Desktop\VSCodes\M.Sc\data\LTest.csv', "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(samples)
work()