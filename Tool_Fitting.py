#Roman Sultanov
#Imports
from AmpLambda import *
#from AmpXi import *
Px, Py, Pz = 0.0, 0.0, 0.95
HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i \
    =0.29, 0.04, -0.16, 1.5, -6.8, 3.1, -13, 4.5, 1.19, -1.03, -3.1, -3.3, -0.7, -4.2

from numba import set_num_threads
set_num_threads(6)
import pandas as pd

#-----------------------------------------------------------------------------------------------------------------------
import csv
with open('MCData.csv', 'r', encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile)
    m2pKdat = []
    m2Kπdat = []
    cosθpdat = []
    φpdat = []
    χdat = []
    
    for row in reader:
        m2pKdat.append(float(row[0]))
        m2Kπdat.append(float(row[1]))
        cosθpdat.append(float(row[2]))
        φpdat.append(float(row[3]))
        χdat.append(float(row[4]))
    
    m2pKdat=np.array(m2pKdat).astype(np.float32)
    m2Kπdat=np.array(m2Kπdat).astype(np.float32)
    cosθpdat=np.array(cosθpdat).astype(np.float32)
    φpdat=np.array(φpdat).astype(np.float32)
    χdat=np.array(χdat).astype(np.float32)

#-----------------------------------------------------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, nogil=True)
def NLL(Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i):
    nll = 0
    for i in prange(1000000):
        nll += -np.log(PDF(m2pKdat[i], m2Kπdat[i], cosθpdat[i], φpdat[i], χdat[i], Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, 1.0, 0.0, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i))
    print(Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i)
    return nll

#-----------------------------------------------------------------------------------------------------------------------

from iminuit import Minuit
m = Minuit(NLL, Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i)
m.errordef = Minuit.LIKELIHOOD
m.strategy = 2
m.limits["Px"] = (-0.1, 0.1)
m.limits["Py"] = (-0.1, 0.1)
m.limits["Pz"] = (-1.0, 1.0)
m.limits["HRpr"] = (-10.0, 10.0)
m.limits["HRpi"] = (-10.0, 10.0)
m.limits["HRmr"] = (-10.0, 10.0)
m.limits["HRmi"] = (-10.0, 10.0)
m.limits["HSpr"] = (-10.0, 10.0)
m.limits["HSpi"] = (-10.0, 10.0)
m.limits["HSmr"] = (-13.5, 13.5)
m.limits["HSmi"] = (-10.0, 10.0)
m.limits["HUpmr"] = (-10.0, 10.0)
m.limits["HUpmi"] = (-10.0, 10.0)
m.limits["HUmpr"] = (-10.0, 10.0)
m.limits["HUmpi"] = (-10.0, 10.0)
m.limits["HUm0r"] = (-10.0, 10.0)
m.limits["HUm0i"] = (-10.0, 10.0)
m.migrad()
m.hesse()

#-----------------------------------------------------------------------------------------------------------------------

np.set_printoptions(precision=19)
VAL=np.array(m.values)
ERR=np.array(m.errors)
COVMAT=np.array(m.covariance)

with open('result.txt', "w") as file:    
    file.write("Result:\n")
    file.write(f"[{', '.join(f'{x:.19f}' for x in VAL)}]\n\n")

    file.write("Error:\n")
    file.write(f"[{', '.join(f'{x:.19f}' for x in ERR)}]\n\n")

    file.write("CovMat:\n")
    file.write("[\n")
    for row in COVMAT:
        file.write(f"  [{', '.join(f'{x:.19f}' for x in row)}],\n")
    file.write("]\n") 
