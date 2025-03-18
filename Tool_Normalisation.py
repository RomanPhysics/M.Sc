#Roman Sultanov
#Imports
from Kinematics import *

#-----------------------------------------------------------------------------------------------------------------------
#Data (GeV)
mΛ=2.4677  #Xi_c^+ mass
mp=0.9383  #proton mass
mK=0.4937  #kaon- mass
mπ=0.1396  #pion+ mass
mR=1.519   #Lambda(1520) peak mass
ΓR=0.016   #Lambda(1520) width
mS=1.232   #Delta++(1232) peak mass
ΓS=0.117   #Delta++(1232) width
mU=0.896   #K*(892) peak mass
ΓU=0.047   #K*(892) width
dr = 1.5
di = 5
#-----------------------------------------------------------------------------------------------------------------------

@njit
def DecayrateDist(m2pK, m2Kπ, cosθp, φp, χ):
    Es2 = (m2pK - mp ** 2 + mK ** 2) / (2 * np.sqrt(m2pK))
    Es3 = (mΛ ** 2 - m2pK - mπ ** 2) / (2 * np.sqrt(m2pK))
    sqrt_term1 = np.sqrt(np.maximum(Es2 ** 2 - mK ** 2, 0))
    sqrt_term2 = np.sqrt(np.maximum(Es3 ** 2 - mπ ** 2, 0))
    sq_term = (Es2 + Es3) ** 2
    if m2pK <= (mp + mK) ** 2 or m2pK >= (mΛ - mπ) ** 2 or m2Kπ <= sq_term - (sqrt_term1 + sqrt_term2) ** 2 or m2Kπ >= sq_term - (sqrt_term1 - sqrt_term2) ** 2:
        return 0.0

    P1, P2, P3 = FinalStateMomenta(m2pK, m2Kπ, mΛ, mp, mK, mπ)
    θR, θR1, aR, θS, θS1, aS, θbU2 = FinalStateAngles(P1, P2, P3)

    WdR_12pp=np.cos(θR/2)
    WdR_12pm=-np.sin(θR/2)
    WdR_12mp=np.sin(θR/2)
    WdR_12mm=np.cos(θR/2)
    WdR1_32pp=0.5*(3*np.cos(θR1)-1)*np.cos(θR1/2)
    WdR1_32pm=-0.5*(3*np.cos(θR1)+1)*np.sin(θR1/2)
    WdR1_32mp=0.5*(3*np.cos(θR1)+1)*np.sin(θR1/2)
    WdR1_32mm=0.5*(3*np.cos(θR1)-1)*np.cos(θR1/2)
    AR=θR+θR1+aR
    WdAR_12pp=np.cos(AR/2)
    WdAR_12pm=-np.sin(AR/2)
    WdAR_12mp=np.sin(AR/2)
    WdAR_12mm=np.cos(AR/2)

    pmR = TwoBodyMomenta(np.sqrt(m2pK), mp, mK)
    p0R = TwoBodyMomenta(mR, mp, mK)
    qmR = TwoBodyMomenta(mΛ, np.sqrt(m2pK), mπ)
    q0R = TwoBodyMomenta(mΛ, mR, mπ)
    FrR = np.sqrt((9 + 3 * (p0R * dr) ** 2 + (p0R * dr) ** 4) / (9 + 3 * (pmR * dr) ** 2 + (pmR * dr) ** 4))
    ΓmR = ΓR * (pmR / p0R) ** 5 * (mR / np.sqrt(m2pK)) * (FrR) ** 2
    BWR=(qmR / q0R) * (pmR / p0R) ** 2 * np.sqrt((1 + (q0R * di) ** 2) / (1 + (qmR * di) ** 2)) * FrR / \
        (mR ** 2 - m2pK - ΓmR * mR * 1j)

    WdS_12pp=np.cos(θS/2)
    WdS_12pm=-np.sin(θS/2)
    WdS_12mp=np.sin(θS/2)
    WdS_12mm=np.cos(θS/2)
    WdS1_32pp=0.5*(3*np.cos(θS1)-1)*np.cos(θS1/2)
    WdS1_32pm=-0.5*(3*np.cos(θS1)+1)*np.sin(θS1/2)
    WdS1_32mp=0.5*(3*np.cos(θS1)+1)*np.sin(θS1/2)
    WdS1_32mm=0.5*(3*np.cos(θS1)-1)*np.cos(θS1/2)
    AS=θS+θS1-aS
    WdAS_12pp=np.cos(AS/2)
    WdAS_12pm=-np.sin(AS/2)
    WdAS_12mp=np.sin(AS/2)
    WdAS_12mm=np.cos(AS/2)

    m2=mΛ**2 + mp**2 + mK**2 + mπ**2 - m2pK - m2Kπ
    pmS = TwoBodyMomenta(np.sqrt(m2), mp, mπ)
    p0S = TwoBodyMomenta(mS, mp, mπ)
    qmS = TwoBodyMomenta(mΛ, np.sqrt(m2), mK)
    q0S = TwoBodyMomenta(mΛ, mS, mK)
    FrS = np.sqrt((1 + (p0S * dr) ** 2) / (1 + (pmS * dr) ** 2))
    ΓmS = ΓS * (pmS / p0S) ** 3 * (mS / np.sqrt(m2)) * (FrS) ** 2
    BWS = (qmS / q0S) * (pmS / p0S) * np.sqrt((1 + (q0S * di) ** 2) / (1 + (qmS * di) ** 2)) * FrS / \
          (mS ** 2 - m2 - ΓmS * mS * 1j)

    WdbU2_1p0=-1/np.sqrt(2)*np.sin(θbU2)
    WdbU2_100=np.cos(θbU2)
    WdbU2_1m0=1/np.sqrt(2)*np.sin(θbU2)

    pmU = TwoBodyMomenta(np.sqrt(m2Kπ), mK, mπ)
    p0U = TwoBodyMomenta(mU, mK, mπ)
    FrU = np.sqrt((1 + (p0U * dr) ** 2) / (1 + (pmU * dr) ** 2))
    ΓmU = ΓU * (pmU / p0U) **3 * (mU / np.sqrt(m2Kπ)) * (FrU) ** 2
    BWU = (pmU / p0U) * FrU / (mU ** 2 - m2Kπ - ΓmU * mU * 1j)

    θp = np.arccos(cosθp)
    cWDE_12pp=np.exp(1j/2*φp)*np.cos(θp/2)*np.exp(1j/2*χ)
    cWDE_12pm=-np.exp(1j/2*φp)*np.sin(θp/2)*np.exp(-1j/2*χ)
    cWDE_12mp=np.exp(-1j/2*φp)*np.sin(θp/2)*np.exp(1j/2*χ)
    cWDE_12mm=np.exp(-1j/2*φp)*np.cos(θp/2)*np.exp(-1j/2*χ)

    ppC1=(WdAR_12pp*WdR_12pp*WdR1_32pp*BWR-WdAR_12pm*WdR_12pp*WdR1_32pm*BWR)*cWDE_12pp+(WdAR_12pp*WdR_12mp*WdR1_32pp*BWR-WdAR_12pm*WdR_12mp*WdR1_32pm*BWR)*cWDE_12pm
    ppC2=(WdAR_12pp*WdR_12pm*WdR1_32mp*BWR-WdAR_12pm*WdR_12pm*WdR1_32mm*BWR)*cWDE_12pp+(WdAR_12pp*WdR_12mm*WdR1_32mp*BWR-WdAR_12pm*WdR_12mm*WdR1_32mm*BWR)*cWDE_12pm
    ppC3=(WdAS_12pp*WdS_12pp*WdS1_32pp*BWS+WdAS_12pm*WdS_12pp*WdS1_32pm*BWS)*cWDE_12pp+(WdAS_12pp*WdS_12mp*WdS1_32pp*BWS+WdAS_12pm*WdS_12mp*WdS1_32pm*BWS)*cWDE_12pm
    ppC4=(WdAS_12pp*WdS_12pm*WdS1_32mp*BWS+WdAS_12pm*WdS_12pm*WdS1_32mm*BWS)*cWDE_12pp+(WdAS_12pp*WdS_12mm*WdS1_32mp*BWS+WdAS_12pm*WdS_12mm*WdS1_32mm*BWS)*cWDE_12pm
    ppC5=WdbU2_100*cWDE_12pp*BWU
    ppC6=WdbU2_1m0*cWDE_12pm*BWU
    ppC7=0
    ppC8=0

    pmC1=(WdAR_12mp*WdR_12pp*WdR1_32pp*BWR-WdAR_12mm*WdR_12pp*WdR1_32pm*BWR)*cWDE_12pp+(WdAR_12mp*WdR_12mp*WdR1_32pp*BWR-WdAR_12mm*WdR_12mp*WdR1_32pm*BWR)*cWDE_12pm
    pmC2=(WdAR_12mp*WdR_12pm*WdR1_32mp*BWR-WdAR_12mm*WdR_12pm*WdR1_32mm*BWR)*cWDE_12pp+(WdAR_12mp*WdR_12mm*WdR1_32mp*BWR-WdAR_12mm*WdR_12mm*WdR1_32mm*BWR)*cWDE_12pm
    pmC3=(WdAS_12mp*WdS_12pp*WdS1_32pp*BWS+WdAS_12mm*WdS_12pp*WdS1_32pm*BWS)*cWDE_12pp+(WdAS_12mp*WdS_12mp*WdS1_32pp*BWS+WdAS_12mm*WdS_12mp*WdS1_32pm*BWS)*cWDE_12pm
    pmC4=(WdAS_12mp*WdS_12pm*WdS1_32mp*BWS+WdAS_12mm*WdS_12pm*WdS1_32mm*BWS)*cWDE_12pp+(WdAS_12mp*WdS_12mm*WdS1_32mp*BWS+WdAS_12mm*WdS_12mm*WdS1_32mm*BWS)*cWDE_12pm
    pmC5=0
    pmC6=0
    pmC7=WdbU2_1p0*cWDE_12pp*BWU
    pmC8=WdbU2_100*cWDE_12pm*BWU

    mpC1=(WdAR_12pp*WdR_12pp*WdR1_32pp*BWR-WdAR_12pm*WdR_12pp*WdR1_32pm*BWR)*cWDE_12mp+(WdAR_12pp*WdR_12mp*WdR1_32pp*BWR-WdAR_12pm*WdR_12mp*WdR1_32pm*BWR)*cWDE_12mm
    mpC2=(WdAR_12pp*WdR_12pm*WdR1_32mp*BWR-WdAR_12pm*WdR_12pm*WdR1_32mm*BWR)*cWDE_12mp+(WdAR_12pp*WdR_12mm*WdR1_32mp*BWR-WdAR_12pm*WdR_12mm*WdR1_32mm*BWR)*cWDE_12mm
    mpC3=(WdAS_12pp*WdS_12pp*WdS1_32pp*BWS+WdAS_12pm*WdS_12pp*WdS1_32pm*BWS)*cWDE_12mp+(WdAS_12pp*WdS_12mp*WdS1_32pp*BWS+WdAS_12pm*WdS_12mp*WdS1_32pm*BWS)*cWDE_12mm
    mpC4=(WdAS_12pp*WdS_12pm*WdS1_32mp*BWS+WdAS_12pm*WdS_12pm*WdS1_32mm*BWS)*cWDE_12mp+(WdAS_12pp*WdS_12mm*WdS1_32mp*BWS+WdAS_12pm*WdS_12mm*WdS1_32mm*BWS)*cWDE_12mm
    mpC5=WdbU2_100*cWDE_12mp*BWU
    mpC6=WdbU2_1m0*cWDE_12mm*BWU
    mpC7=0
    mpC8=0

    mmC1=(WdAR_12mp*WdR_12pp*WdR1_32pp*BWR-WdAR_12mm*WdR_12pp*WdR1_32pm*BWR)*cWDE_12mp+(WdAR_12mp*WdR_12mp*WdR1_32pp*BWR-WdAR_12mm*WdR_12mp*WdR1_32pm*BWR)*cWDE_12mm
    mmC2=(WdAR_12mp*WdR_12pm*WdR1_32mp*BWR-WdAR_12mm*WdR_12pm*WdR1_32mm*BWR)*cWDE_12mp+(WdAR_12mp*WdR_12mm*WdR1_32mp*BWR-WdAR_12mm*WdR_12mm*WdR1_32mm*BWR)*cWDE_12mm
    mmC3=(WdAS_12mp*WdS_12pp*WdS1_32pp*BWS+WdAS_12mm*WdS_12pp*WdS1_32pm*BWS)*cWDE_12mp+(WdAS_12mp*WdS_12mp*WdS1_32pp*BWS+WdAS_12mm*WdS_12mp*WdS1_32pm*BWS)*cWDE_12mm
    mmC4=(WdAS_12mp*WdS_12pm*WdS1_32mp*BWS+WdAS_12mm*WdS_12pm*WdS1_32mm*BWS)*cWDE_12mp+(WdAS_12mp*WdS_12mm*WdS1_32mp*BWS+WdAS_12mm*WdS_12mm*WdS1_32mm*BWS)*cWDE_12mm
    mmC5=0
    mmC6=0
    mmC7=WdbU2_1p0*cWDE_12mp*BWU
    mmC8=WdbU2_100*cWDE_12mm*BWU

    return np.real(ppC1*np.conjugate(ppC1) + pmC1*np.conjugate(pmC1) + mpC1*np.conjugate(mpC1) + mmC1*np.conjugate(mmC1))


import vegas
@vegas.lbatchintegrand
def I(x):
    return np.array([DecayrateDist(x1, x2, x3, x4, x5) for x1, x2, x3, x4, x5 in zip(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4])])


def main():
    import time
    start_time = time.time()
    integ = vegas.Integrator([[(mp + mK)**2, (mΛ - mπ)**2], [(mπ + mK)**2, (mΛ - mp)**2], [-1.0, 1.0], [-np.pi, np.pi], [-np.pi, np.pi]], nproc=6)
    integ(I, nitn=10, neval=30000000)
    intt=integ(I, nitn=15, neval=30000000)
    print(intt.summary())
    print(intt.mean)
    end_time = time.time()
    totalseconds=end_time - start_time
    print(f"Time taken: {totalseconds//60} m, {totalseconds%60} s.")

if __name__ == '__main__':
    main()