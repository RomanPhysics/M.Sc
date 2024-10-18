#Roman Sultanov
#Imports
from Form import *

#-----------------------------------------------------------------------------------------------------------------------
#Data (GeV)
mΛ=2.2865  #lambda_c^+ mass
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
def DecayrateDist(m2pK, m2Kπ, cosθp, φp, χ, Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUp0r, HUp0i, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i):
    #Es2 = (m2pK - mp ** 2 + mK ** 2) / (2 * np.sqrt(m2pK))
    #Es3 = (mΛ ** 2 - m2pK - mπ ** 2) / (2 * np.sqrt(m2pK))
    #sqrt_term1 = np.sqrt(np.maximum(Es2 ** 2 - mK ** 2, 0))
    #sqrt_term2 = np.sqrt(np.maximum(Es3 ** 2 - mπ ** 2, 0))
    #sq_term = (Es2 + Es3) ** 2
    #if m2pK <= (mp + mK) ** 2 or m2pK >= (mΛ - mπ) ** 2 or m2Kπ <= sq_term - (sqrt_term1 + sqrt_term2) ** 2 or m2Kπ >= sq_term - (sqrt_term1 - sqrt_term2) ** 2:
    #    return 0.0

    P1, P2, P3 = FinalStateMomenta(m2pK, m2Kπ, cosθp, φp, χ, mΛ, mp, mK, mπ)
    φR, θR, φR1, θR1, αR1, βR1, γR1, φS, θS, φS1, θS1, αS1, βS1, γS1, φ1, θ1, φbU2, θbU2 = FinalStateAngles(P1, P2, P3)

    HRp=HRpr+HRpi*1j
    HRm=HRmr+HRmi*1j
    HSp=HSpr+HSpi*1j
    HSm=HSmr+HSmi*1j
    HUpm=HUpmr+HUpmi*1j
    HUp0=HUp0r+HUp0i*1j
    HUmp=HUmpr+HUmpi*1j
    HUm0=HUm0r+HUm0i*1j

    HRpC=np.conjugate(HRp)
    HRmC=np.conjugate(HRm)
    HSpC=np.conjugate(HSp)
    HSmC=np.conjugate(HSm)
    HUp0C=np.conjugate(HUp0)
    HUpmC=np.conjugate(HUpm)
    HUmpC=np.conjugate(HUmp)
    HUm0C=np.conjugate(HUm0)

    cWDR1_32pm=-0.5*(3*np.cos(θR1)+1)*np.sin(θR1/2)*np.exp(φR1*1j/2)
    cWDR1_32mp=0.5*(3*np.cos(θR1)+1)*np.sin(θR1/2)*np.exp(-φR1*1j/2)
    cWDR1_32pp=0.5*(3*np.cos(θR1)-1)*np.cos(θR1/2)*np.exp(φR1*1j/2)
    cWDR1_32mm=0.5*(3*np.cos(θR1)-1)*np.cos(θR1/2)*np.exp(-φR1*1j/2)
    cWDR_12pm=-np.sin(θR/2)*np.exp(φR*1j/2)
    cWDR_12mp=np.sin(θR/2)*np.exp(-φR*1j/2)
    cWDR_12pp=np.cos(θR/2)*np.exp(φR*1j/2)
    cWDR_12mm=np.cos(θR/2)*np.exp(-φR*1j/2)
    WDR1_12mm=np.cos(θR1/2)*np.exp(φR1*1j/2)
    WDR1_12pm=-np.sin(θR1/2)*np.exp(-φR1*1j/2)
    WDR1_12mp=np.sin(θR1/2)*np.exp(φR1*1j/2)
    WDR1_12pp=np.cos(θR1/2)*np.exp(-φR1*1j/2)
    WDR_12mm=np.conjugate(cWDR_12mm)
    WDR_12pm=np.conjugate(cWDR_12pm)
    WDR_12mp=np.conjugate(cWDR_12mp)
    WDR_12pp=np.conjugate(cWDR_12pp)
    WDwR_12pp=np.cos(βR1/2)*np.exp(-(αR1+γR1)*1j/2)
    WDwR_12pm=np.sin(-βR1/2)*np.exp((-αR1 + γR1)*1j/2)
    WDwR_12mp=np.sin(βR1/2)*np.exp((αR1 - γR1)*1j/2)
    WDwR_12mm=np.cos(βR1/2)*np.exp((αR1+γR1)*1j/2)

    pmR = TwoBodyMomenta(np.sqrt(m2pK), mp, mK)
    p0R = TwoBodyMomenta(mR, mp, mK)
    qmR = TwoBodyMomenta(mΛ, np.sqrt(m2pK), mπ)
    q0R = TwoBodyMomenta(mΛ, mR, mπ)
    FrR = np.sqrt((9 + 3 * (p0R * dr) ** 2 + (p0R * dr) ** 4) / (9 + 3 * (pmR * dr) ** 2 + (pmR * dr) ** 4))
    ΓmR = ΓR * (pmR / p0R) ** 5 * (mR / np.sqrt(m2pK)) * (FrR) ** 2
    bwr=(qmR / q0R) * (pmR / p0R) ** 2 * np.sqrt((1 + (q0R * di) ** 2) / (1 + (qmR * di) ** 2)) * FrR / \
        (mR ** 2 - m2pK - ΓmR * mR * 1j)

    cWDS1_32pm=-0.5*(3*np.cos(θS1)+1)*np.sin(θS1/2)*np.exp(φS1*1j/2)
    cWDS1_32mp=0.5*(3*np.cos(θS1)+1)*np.sin(θS1/2)*np.exp(-φS1*1j/2)
    cWDS1_32pp=0.5*(3*np.cos(θS1)-1)*np.cos(θS1/2)*np.exp(φS1*1j/2)
    cWDS1_32mm=0.5*(3*np.cos(θS1)-1)*np.cos(θS1/2)*np.exp(-φS1*1j/2)
    cWDS_12pm=-np.sin(θS/2)*np.exp(φS*1j/2)
    cWDS_12mp=np.sin(θS/2)*np.exp(-φS*1j/2)
    cWDS_12pp=np.cos(θS/2)*np.exp(φS*1j/2)
    cWDS_12mm=np.cos(θS/2)*np.exp(-φS*1j/2)
    WDS1_12mm=np.cos(θS1/2)*np.exp(φS1*1j/2)
    WDS1_12pm=-np.sin(θS1/2)*np.exp(-φS1*1j/2)
    WDS1_12mp=np.sin(θS1/2)*np.exp(φS1*1j/2)
    WDS1_12pp=np.cos(θS1/2)*np.exp(-φS1*1j/2)
    WDS_12mm=np.conjugate(cWDS_12mm)
    WDS_12pm=np.conjugate(cWDS_12pm)
    WDS_12mp=np.conjugate(cWDS_12mp)
    WDS_12pp=np.conjugate(cWDS_12pp)
    WDwS_12pp=np.cos(βS1/2)*np.exp(-(αS1+γS1)*1j/2)
    WDwS_12pm=np.sin(-βS1/2)*np.exp((-αS1 + γS1)*1j/2)
    WDwS_12mp=np.sin(βS1/2)*np.exp((αS1 - γS1)*1j/2)
    WDwS_12mm=np.cos(βS1/2)*np.exp((αS1+γS1)*1j/2)

    m2=mΛ**2 + mp**2 + mK**2 + mπ**2 - m2pK - m2Kπ
    pmS = TwoBodyMomenta(np.sqrt(m2), mp, mπ)
    p0S = TwoBodyMomenta(mS, mp, mπ)
    qmS = TwoBodyMomenta(mΛ, np.sqrt(m2), mK)
    q0S = TwoBodyMomenta(mΛ, mS, mK)
    FrS = np.sqrt((1 + (p0S * dr) ** 2) / (1 + (pmS * dr) ** 2))
    ΓmS = ΓS * (pmS / p0S) ** 3 * (mS / np.sqrt(m2)) * (FrS) ** 2
    bws = (qmS / q0S) * (pmS / p0S) * np.sqrt((1 + (q0S * di) ** 2) / (1 + (qmS * di) ** 2)) * FrS / \
          (mS ** 2 - m2 - ΓmS * mS * 1j)

    cWD1_12pp=np.cos(θ1/2)*np.exp(φ1*1j/2)
    cWD1_12mp=np.sin(θ1/2)*np.exp(-φ1*1j/2)
    cWD1_12pm=-np.sin(θ1/2)*np.exp(φ1*1j/2)
    cWD1_12mm=np.cos(θ1/2)*np.exp(-φ1*1j/2)
    cWDbU2_1p0=-1/np.sqrt(2)*np.sin(θbU2)*np.exp(φbU2*1j)
    cWDbU2_1m0=1/np.sqrt(2)*np.sin(θbU2)*np.exp(-φbU2*1j)
    cWDbU2_100=np.cos(θbU2)
    WD1_12pm=np.conjugate(cWD1_12pm)
    WD1_12pp=np.conjugate(cWD1_12pp)
    WD1_12mp=np.conjugate(cWD1_12mp)
    WD1_12mm=np.conjugate(cWD1_12mm)

    pmU = TwoBodyMomenta(np.sqrt(m2Kπ), mK, mπ)
    p0U = TwoBodyMomenta(mU, mK, mπ)
    FrU = np.sqrt((1 + (p0U * dr) ** 2) / (1 + (pmU * dr) ** 2))
    ΓmU = ΓU * (pmU / p0U) **3 * (mU / np.sqrt(m2Kπ)) * (FrU) ** 2
    bwu = (pmU / p0U) * FrU / (mU ** 2 - m2Kπ - ΓmU * mU * 1j)

    ppC1=-cWDR_12pp*cWDR1_32pm*(WDwR_12pm*WDR_12mm*WDR1_12mm+WDwR_12pp*WDR_12pm*WDR1_12mm+WDwR_12pm*WDR_12mp*WDR1_12pm+WDwR_12pp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12pp*cWDR1_32pp*(WDwR_12pm*WDR_12mm*WDR1_12mp+WDwR_12pp*WDR_12pm*WDR1_12mp+WDwR_12pm*WDR_12mp*WDR1_12pp+WDwR_12pp*WDR_12pp*WDR1_12pp)*bwr
    ppC2=-cWDR_12pm*cWDR1_32mm*(WDwR_12pm*WDR_12mm*WDR1_12mm+WDwR_12pp*WDR_12pm*WDR1_12mm+WDwR_12pm*WDR_12mp*WDR1_12pm+WDwR_12pp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12pm*cWDR1_32mp*(WDwR_12pm*WDR_12mm*WDR1_12mp+WDwR_12pp*WDR_12pm*WDR1_12mp+WDwR_12pm*WDR_12mp*WDR1_12pp+WDwR_12pp*WDR_12pp*WDR1_12pp)*bwr
    ppC3=cWDS_12pp*cWDS1_32pm*(WDwS_12pm*WDS_12mm*WDS1_12mm+WDwS_12pp*WDS_12pm*WDS1_12mm+WDwS_12pm*WDS_12mp*WDS1_12pm+WDwS_12pp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12pp*cWDS1_32pp*(WDwS_12pm*WDS_12mm*WDS1_12mp+WDwS_12pp*WDS_12pm*WDS1_12mp+WDwS_12pm*WDS_12mp*WDS1_12pp+WDwS_12pp*WDS_12pp*WDS1_12pp)*bws
    ppC4=cWDS_12pm*cWDS1_32mm*(WDwS_12pm*WDS_12mm*WDS1_12mm+WDwS_12pp*WDS_12pm*WDS1_12mm+WDwS_12pm*WDS_12mp*WDS1_12pm+WDwS_12pp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12pm*cWDS1_32mp*(WDwS_12pm*WDS_12mm*WDS1_12mp+WDwS_12pp*WDS_12pm*WDS1_12mp+WDwS_12pm*WDS_12mp*WDS1_12pp+WDwS_12pp*WDS_12pp*WDS1_12pp)*bws
    ppC5=cWD1_12pp*cWDbU2_100*WD1_12pp*bwu
    ppC6=cWD1_12pm*cWDbU2_1m0*WD1_12pp*bwu
    ppC7=cWD1_12pp*cWDbU2_1p0*WD1_12pm*bwu
    ppC8=cWD1_12pm*cWDbU2_100*WD1_12pm*bwu

    App = HRp*ppC1 + HRm*ppC2 +HSp*ppC3 + HSm*ppC4 + HUp0*ppC5 + HUpm*ppC6 + HUmp*ppC7 + HUm0*ppC8

    pmC1=-cWDR_12pp*cWDR1_32pm*(WDwR_12mm*WDR_12mm*WDR1_12mm+WDwR_12mp*WDR_12pm*WDR1_12mm+WDwR_12mm*WDR_12mp*WDR1_12pm+WDwR_12mp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12pp*cWDR1_32pp*(WDwR_12mm*WDR_12mm*WDR1_12mp+WDwR_12mp*WDR_12pm*WDR1_12mp+WDwR_12mm*WDR_12mp*WDR1_12pp+WDwR_12mp*WDR_12pp*WDR1_12pp)*bwr
    pmC2=-cWDR_12pm*cWDR1_32mm*(WDwR_12mm*WDR_12mm*WDR1_12mm+WDwR_12mp*WDR_12pm*WDR1_12mm+WDwR_12mm*WDR_12mp*WDR1_12pm+WDwR_12mp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12pm*cWDR1_32mp*(WDwR_12mm*WDR_12mm*WDR1_12mp+WDwR_12mp*WDR_12pm*WDR1_12mp+WDwR_12mm*WDR_12mp*WDR1_12pp+WDwR_12mp*WDR_12pp*WDR1_12pp)*bwr
    pmC3=cWDS_12pp*cWDS1_32pm*(WDwS_12mm*WDS_12mm*WDS1_12mm+WDwS_12mp*WDS_12pm*WDS1_12mm+WDwS_12mm*WDS_12mp*WDS1_12pm+WDwS_12mp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12pp*cWDS1_32pp*(WDwS_12mm*WDS_12mm*WDS1_12mp+WDwS_12mp*WDS_12pm*WDS1_12mp+WDwS_12mm*WDS_12mp*WDS1_12pp+WDwS_12mp*WDS_12pp*WDS1_12pp)*bws
    pmC4=cWDS_12pm*cWDS1_32mm*(WDwS_12mm*WDS_12mm*WDS1_12mm+WDwS_12mp*WDS_12pm*WDS1_12mm+WDwS_12mm*WDS_12mp*WDS1_12pm+WDwS_12mp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12pm*cWDS1_32mp*(WDwS_12mm*WDS_12mm*WDS1_12mp+WDwS_12mp*WDS_12pm*WDS1_12mp+WDwS_12mm*WDS_12mp*WDS1_12pp+WDwS_12mp*WDS_12pp*WDS1_12pp)*bws
    pmC5=cWD1_12pp*cWDbU2_100*WD1_12mp*bwu
    pmC6=cWD1_12pm*cWDbU2_1m0*WD1_12mp*bwu
    pmC7=cWD1_12pp*cWDbU2_1p0*WD1_12mm*bwu
    pmC8=cWD1_12pm*cWDbU2_100*WD1_12mm*bwu

    Apm = HRp*pmC1 + HRm*pmC2 +HSp*pmC3 + HSm*pmC4 + HUp0*pmC5 + HUpm*pmC6 + HUmp*pmC7 + HUm0*pmC8

    mpC1=-cWDR_12mp*cWDR1_32pm*(WDwR_12pm*WDR_12mm*WDR1_12mm+WDwR_12pp*WDR_12pm*WDR1_12mm+WDwR_12pm*WDR_12mp*WDR1_12pm+WDwR_12pp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12mp*cWDR1_32pp*(WDwR_12pm*WDR_12mm*WDR1_12mp+WDwR_12pp*WDR_12pm*WDR1_12mp+WDwR_12pm*WDR_12mp*WDR1_12pp+WDwR_12pp*WDR_12pp*WDR1_12pp)*bwr
    mpC2=-cWDR_12mm*cWDR1_32mm*(WDwR_12pm*WDR_12mm*WDR1_12mm+WDwR_12pp*WDR_12pm*WDR1_12mm+WDwR_12pm*WDR_12mp*WDR1_12pm+WDwR_12pp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12mm*cWDR1_32mp*(WDwR_12pm*WDR_12mm*WDR1_12mp+WDwR_12pp*WDR_12pm*WDR1_12mp+WDwR_12pm*WDR_12mp*WDR1_12pp+WDwR_12pp*WDR_12pp*WDR1_12pp)*bwr
    mpC3=cWDS_12mp*cWDS1_32pm*(WDwS_12pm*WDS_12mm*WDS1_12mm+WDwS_12pp*WDS_12pm*WDS1_12mm+WDwS_12pm*WDS_12mp*WDS1_12pm+WDwS_12pp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12mp*cWDS1_32pp*(WDwS_12pm*WDS_12mm*WDS1_12mp+WDwS_12pp*WDS_12pm*WDS1_12mp+WDwS_12pm*WDS_12mp*WDS1_12pp+WDwS_12pp*WDS_12pp*WDS1_12pp)*bws
    mpC4=cWDS_12mm*cWDS1_32mm*(WDwS_12pm*WDS_12mm*WDS1_12mm+WDwS_12pp*WDS_12pm*WDS1_12mm+WDwS_12pm*WDS_12mp*WDS1_12pm+WDwS_12pp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12mm*cWDS1_32mp*(WDwS_12pm*WDS_12mm*WDS1_12mp+WDwS_12pp*WDS_12pm*WDS1_12mp+WDwS_12pm*WDS_12mp*WDS1_12pp+WDwS_12pp*WDS_12pp*WDS1_12pp)*bws
    mpC5=cWD1_12mp*cWDbU2_100*WD1_12pp*bwu
    mpC6=cWD1_12mm*cWDbU2_1m0*WD1_12pp*bwu
    mpC7=cWD1_12mp*cWDbU2_1p0*WD1_12pm*bwu
    mpC8=cWD1_12mm*cWDbU2_100*WD1_12pm*bwu

    Amp = HRp*mpC1 + HRm*mpC2 +HSp*mpC3 + HSm*mpC4 + HUp0*mpC5 + HUpm*mpC6 + HUmp*mpC7 + HUm0*mpC8

    mmC1=-cWDR_12mp*cWDR1_32pm*(WDwR_12mm*WDR_12mm*WDR1_12mm+WDwR_12mp*WDR_12pm*WDR1_12mm+WDwR_12mm*WDR_12mp*WDR1_12pm+WDwR_12mp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12mp*cWDR1_32pp*(WDwR_12mm*WDR_12mm*WDR1_12mp+WDwR_12mp*WDR_12pm*WDR1_12mp+WDwR_12mm*WDR_12mp*WDR1_12pp+WDwR_12mp*WDR_12pp*WDR1_12pp)*bwr
    mmC2=-cWDR_12mm*cWDR1_32mm*(WDwR_12mm*WDR_12mm*WDR1_12mm+WDwR_12mp*WDR_12pm*WDR1_12mm+WDwR_12mm*WDR_12mp*WDR1_12pm+WDwR_12mp*WDR_12pp*WDR1_12pm)*bwr\
        +cWDR_12mm*cWDR1_32mp*(WDwR_12mm*WDR_12mm*WDR1_12mp+WDwR_12mp*WDR_12pm*WDR1_12mp+WDwR_12mm*WDR_12mp*WDR1_12pp+WDwR_12mp*WDR_12pp*WDR1_12pp)*bwr
    mmC3=cWDS_12mp*cWDS1_32pm*(WDwS_12mm*WDS_12mm*WDS1_12mm+WDwS_12mp*WDS_12pm*WDS1_12mm+WDwS_12mm*WDS_12mp*WDS1_12pm+WDwS_12mp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12mp*cWDS1_32pp*(WDwS_12mm*WDS_12mm*WDS1_12mp+WDwS_12mp*WDS_12pm*WDS1_12mp+WDwS_12mm*WDS_12mp*WDS1_12pp+WDwS_12mp*WDS_12pp*WDS1_12pp)*bws
    mmC4=cWDS_12mm*cWDS1_32mm*(WDwS_12mm*WDS_12mm*WDS1_12mm+WDwS_12mp*WDS_12pm*WDS1_12mm+WDwS_12mm*WDS_12mp*WDS1_12pm+WDwS_12mp*WDS_12pp*WDS1_12pm)*bws\
         +cWDS_12mm*cWDS1_32mp*(WDwS_12mm*WDS_12mm*WDS1_12mp+WDwS_12mp*WDS_12pm*WDS1_12mp+WDwS_12mm*WDS_12mp*WDS1_12pp+WDwS_12mp*WDS_12pp*WDS1_12pp)*bws
    mmC5=cWD1_12mp*cWDbU2_100*WD1_12mp*bwu
    mmC6=cWD1_12mm*cWDbU2_1m0*WD1_12mp*bwu
    mmC7=cWD1_12mp*cWDbU2_1p0*WD1_12mm*bwu
    mmC8=cWD1_12mm*cWDbU2_100*WD1_12mm*bwu

    Amm = HRp*mmC1 + HRm*mmC2 +HSp*mmC3 + HSm*mmC4 + HUp0*mmC5 + HUpm*mmC6 + HUmp*mmC7 + HUm0*mmC8


    NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2=HRp*HRpC*(5311.814872048573)                        +HRm*HRpC*(0)                                       +HSp*HRpC*(204.6937609247803-6.007421260172587j)     +HSm*HRpC*(-287.2560475344617+149.64058433328609j)   +HUp0*HRpC*(189.99171355418065-266.8169844880696j)   +HUpm*HRpC*(-172.1689365807526-91.80621696804803j) +HUmp*HRpC*(295.5023133321682-314.48754754001925j) +HUm0*HRpC*(-157.61475492541712-25.851853170771335j)\
                                    +HRp*HRmC*(0)                                        +HRm*HRmC*(5311.814872048573)                       +HSp*HRmC*(287.2560475344617-149.64058433328609j)    +HSm*HRmC*(-204.6937609247803+6.007421260172587j)    +HUp0*HRmC*(157.61475492541712+25.851853170771335j)  +HUpm*HRmC*(-295.5023133321682+314.48754754001925j)+HUmp*HRmC*(172.1689365807526+91.80621696804803j)  +HUm0*HRmC*(-189.99171355418065+266.8169844880696j)\
                                    +HRp*HSpC*(204.6937609247803+6.007421260172587j)     +HRm*HSpC*(287.2560475344617+149.64058433328609j)   +HSp*HSpC*(823.9195265936702)                        +HSm*HSpC*(0)                                        +HUp0*HSpC*(-136.65038380072326+134.79009858281137j) +HUpm*HSpC*(38.64034900132792+74.67096150196075j)  +HUmp*HSpC*(258.85864285874237-57.46888540504888j) +HUm0*HSpC*(-25.181691922845722-40.746441452622705j)\
                                    +HRp*HSmC*(-287.2560475344617-149.64058433328609j)   +HRm*HSmC*(-204.6937609247803-6.007421260172587j)   +HSp*HSmC*(0)                                        +HSm*HSmC*(823.9195265936702)                        +HUp0*HSmC*(-25.181691922845722-40.746441452622705j) +HUpm*HSmC*(258.85864285874237-57.46888540504888j) +HUmp*HSmC*(38.64034900132792+74.67096150196075j)  +HUm0*HSmC*(-136.65038380072326+134.79009858281137j)\
                                    +HRp*HUp0C*(189.99171355418065+266.8169844880696j)   +HRm*HUp0C*(157.61475492541712-25.851853170771335j) +HSp*HUp0C*(-136.65038380072326-134.79009858281137j) +HSm*HUp0C*(-25.181691922845722+40.746441452622705j) +HUp0*HUp0C*(3735.0995605146118)                     +HUpm*HUp0C*(0)                                    +HUmp*HUp0C*(0)                                    +HUm0*HUp0C*(0)\
                                    +HRp*HUpmC*(-172.1689365807526+91.80621696804803j)   +HRm*HUpmC*(-295.5023133321682-314.48754754001925j) +HSp*HUpmC*(38.64034900132792-74.67096150196075j)    +HSm*HUpmC*(258.85864285874237+57.46888540504888j)   +HUp0*HUpmC*(0)                                      +HUpm*HUpmC*(3735.0995605146118)                   +HUmp*HUpmC*(0)                                    +HUm0*HUpmC*(0)\
                                    +HRp*HUmpC*(295.5023133321682+314.48754754001925j)   +HRm*HUmpC*(172.1689365807526-91.80621696804803j)   +HSp*HUmpC*(258.85864285874237+57.46888540504888j)   +HSm*HUmpC*(38.64034900132792-74.67096150196075j)    +HUp0*HUmpC*(0)                                      +HUpm*HUmpC*(0)                                    +HUmp*HUmpC*(3735.0995605146118)                   +HUm0*HUmpC*(0)\
                                    +HRp*HUm0C*(-157.61475492541712+25.851853170771335j) +HRm*HUm0C*(-189.99171355418065-266.8169844880696j) +HSp*HUm0C*(-25.181691922845722+40.746441452622705j) +HSm*HUm0C*(-136.65038380072326-134.79009858281137j) +HUp0*HUm0C*(0)                                      +HUpm*HUm0C*(0)                                    +HUmp*HUm0C*(0)                                    +HUm0*HUm0C*(3735.0995605146118)


    NAppAmpCNApmAmmC=HRp*HRpC*(0.008697670413898602+0.0007811247922970726j)   +HRm*HRpC*(0)                                             +HSp*HRpC*(-0.00201911717393884+0.0016854027651537608j)    +HSm*HRpC*(-0.006090999058410408+0.0035588791213054813j)   +HUp0*HRpC*(-0.00817358573743439-0.0005165821491466046j)    +HUpm*HRpC*(0.0042291054263243625-0.0005373013755476193j)    +HUmp*HRpC*(-0.0028351335214261554-0.0015790975840444565j)   +HUm0*HRpC*(-0.0065220161607247445+0.019936307347001397j)\
                    +HRp*HRmC*(0)                                             +HRm*HRmC*(-0.017624306285781773+0.009191728002932508j)   +HSp*HRmC*(-0.007119418791240112+0.0017118337291957546j)   +HSm*HRmC*(0.0008524332070180355-0.00018100774618937766j)  +HUp0*HRmC*(-0.000972011429526238-0.00006136889017722185j)  +HUpm*HRmC*(-0.008199421595456599+0.0022393512108653973j)    +HUmp*HRmC*(0.0004250497013088334+0.0005412412815609031j)    +HUm0*HRmC*(0.007474326022380617-0.006218321577662965j)\
                    +HRp*HSpC*(-0.0018162717167616669+0.0022765319431453375j) +HRm*HSpC*(0.0031236893282276562-0.0010916951419417801j)  +HSp*HSpC*(-0.004278369786712551+0.0001896572017739734j)   +HSm*HSpC*(0)                                              +HUp0*HSpC*(0.0006428847321675314+0.005572181357708227j)    +HUpm*HSpC*(-0.0010830922069391103-0.0029989647876771685j)   +HUmp*HSpC*(0.000711423266969445+0.002172956878709266j)      +HUm0*HSpC*(-0.002194220008150511-0.0069435972679717526j)\
                    +HRp*HSmC*(0.002548538397814917+0.0013245347320114797j)   +HRm*HSmC*(0.0011500737294786454-0.0011799773509160512j)  +HSp*HSmC*(0)                                              +HSm*HSmC*(0.003773875002698294-0.003873372234565564j)     +HUp0*HSmC*(0.0005477856224695524-0.0014620216648438295j)   +HUpm*HSmC*(-0.002805592566031218+0.003507427073472986j)     +HUmp*HSmC*(0.0008712734575286407-0.0007909711777168737j)    +HUm0*HSmC*(-0.0010105838169701302-0.003007005989834354j)\
                    +HRp*HUp0C*(0.005507733262262957+0.0027003187240321968j)  +HRm*HUp0C*(-0.001249077504150512+0.01166568270523893j)   +HSp*HUp0C*(-0.0026285434242843114+0.0003113075995831372j) +HSm*HUp0C*(0.004366753789239114-0.0020523108368337467j)   +HUp0*HUp0C*(-0.0055349260311695295-0.008084622047917815j)  +HUpm*HUp0C*(0.0042282865997190135-0.015563660734653392j)    +HUmp*HUp0C*(0)                                              +HUm0*HUp0C*(0)\
                    +HRp*HUpmC*(0.00019229187006333814-0.001953991979465638j) +HRm*HUpmC*(-0.009686126184072785-0.00027440540006359236j)+HSp*HUpmC*(-0.002958763783505505-0.001742994480897403j)   +HSm*HUpmC*(-0.0005557239821348546-0.0004969796745332271j) +HUp0*HUpmC*(-0.006353184149850068+0.020205947876991447j)   +HUpm*HUpmC*(-0.0002173825400998003-0.009991750100815334j)   +HUmp*HUpmC*(0)                                              +HUm0*HUpmC*(0)\
                    +HRp*HUmpC*(0.008786930886827676+0.008893971724384574j)   +HRm*HUmpC*(0.003248116475431893+0.0030792078743957386j)  +HSp*HUmpC*(0.002612112009630622+0.00038757370264717j)     +HSm*HUmpC*(-0.0008964252418191892-0.0011641280699403761j) +HUp0*HUmpC*(0)                                             +HUpm*HUmpC*(0)                                              +HUmp*HUmpC*(0.005179590559551519+0.0002995160936085433j)    +HUm0*HUmpC*(0.009947135458512769+0.0034376110871208736j)\
                    +HRp*HUm0C*(0.00616893698245709-0.003471930082332638j)    +HRm*HUm0C*(0.004028339915680458-0.015956413659669773j)   +HSp*HUm0C*(-0.0017202762708192095-0.002717315793663356j)  +HSm*HUm0C*(-0.005152627629858246+0.000810805723327691j)   +HUp0*HUm0C*(0)                                             +HUpm*HUm0C*(0)                                              +HUmp*HUm0C*(0.002268498336240209-0.015674539438528402j)     +HUm0*HUm0C*(-0.002190808275689582-0.0102516967223682j)

    p=(1 + Pz)*np.absolute(App)**2 + (1 - Pz)*np.absolute(Amp)**2 + 2*np.real((Px - Py*1j)*App*np.conjugate(Amp))\
     +(1 + Pz)*np.absolute(Apm)**2 + (1 - Pz)*np.absolute(Amm)**2 + 2*np.real((Px - Py*1j)*Apm*np.conjugate(Amm))

    N = np.real(NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2 + 2*(Px - Py*1j)*NAppAmpCNApmAmmC)

    return p/N