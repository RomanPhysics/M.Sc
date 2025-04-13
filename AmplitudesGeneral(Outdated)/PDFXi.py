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
def PDF(m2pK, m2Kπ, cosθp, φp, χ, Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUp0r, HUp0i, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i):
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

    NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2=HRp*HRpC*(7197.946564963944)                        +HRm*HRpC*(0)                                       +HSp*HRpC*(236.7670697929064+22.268528342668397j)    +HSm*HRpC*(-386.39396887382566+123.91358394887268j)  +HUp0*HRpC*(355.0942217927184-257.6806033247093j)    +HUpm*HRpC*(-132.86840918368028-143.02184952563496j) +HUmp*HRpC*(481.27856470507277-218.45935693926788j)+HUm0*HRpC*(-148.69998641805-67.52115627915572j)\
                                    +HRp*HRmC*(0)                                        +HRm*HRmC*(7197.946564963944)                       +HSp*HRmC*(386.39396887382566-123.91358394887268j)   +HSm*HRmC*(-236.7670697929064-22.267419968178384j)   +HUp0*HRmC*(148.69998641805+67.52115627915572j)      +HUpm*HRmC*(-481.27856470507277+218.45935693926788j) +HUmp*HRmC*(132.86840918368028+143.02184952563496j)+HUm0*HRmC*(-355.0942217927184+257.6806033247093j)\
                                    +HRp*HSpC*(236.7670697929064-22.268528342668397j)    +HRm*HSpC*(386.39396887382566+123.91358394887268j)  +HSp*HSpC*(1119.129511989152)                        +HSm*HSpC*(0)                                        +HUp0*HSpC*(-195.70335603078078+116.34693534292174j) +HUpm*HSpC*(22.108344526359897+78.32155932102214j)   +HUmp*HSpC*(331.82434772472743-24.256420298320936j)+HUm0*HSpC*(-27.344314365657663-43.57121126166499j)\
                                    +HRp*HSmC*(-386.39396887382566-123.91358394887268j)  +HRm*HSmC*(-236.7670697929064+22.267419968178384j)  +HSp*HSmC*(0)                                        +HSm*HSmC*(1119.129511989152)                        +HUp0*HSmC*(-27.344314365657663-43.57121126166499j)  +HUpm*HSmC*(331.82434772472743-24.256420298320936j)  +HUmp*HSmC*(22.108344526359897+78.32155932102214j) +HUm0*HSmC*(-195.70335603078078+116.34693534292174j)\
                                    +HRp*HUp0C*(355.0942217927184+257.6806033247093j)    +HRm*HUp0C*(148.69998641805-67.52115627915572j)     +HSp*HUp0C*(-195.70335603078078-116.34693534292174j) +HSm*HUp0C*(-27.344314365657663+43.57121126166499j)  +HUp0*HUp0C*(4977.908273555728)                      +HUpm*HUp0C*(0)                                      +HUmp*HUp0C*(0)                                    +HUm0*HUp0C*(0)\
                                    +HRp*HUpmC*(-132.86840918368028+143.02184952563496j) +HRm*HUpmC*(-481.27856470507277-218.45935693926788j)+HSp*HUpmC*(22.108344526359897-78.32155932102214j)   +HSm*HUpmC*(331.82434772472743+24.256420298320936j)  +HUp0*HUpmC*(0)                                      +HUpm*HUpmC*(4977.908273555728)                      +HUmp*HUpmC*(0)                                    +HUm0*HUpmC*(0)\
                                    +HRp*HUmpC*(481.27856470507277+218.45935693926788j)  +HRm*HUmpC*(132.86840918368028-143.02184952563496j) +HSp*HUmpC*(331.82434772472743+24.256420298320936j)  +HSm*HUmpC*(22.108344526359897-78.32155932102214j)   +HUp0*HUmpC*(0)                                      +HUpm*HUmpC*(0)                                      +HUmp*HUmpC*(4977.908273555728)                    +HUm0*HUmpC*(0)\
                                    +HRp*HUm0C*(-148.69998641805+67.52115627915572j)     +HRm*HUm0C*(-355.0942217927184-257.6806033247093j)  +HSp*HUm0C*(-27.344314365657663+43.57121126166499j)  +HSm*HUm0C*(-195.70335603078078-116.34693534292174j) +HUp0*HUm0C*(0)                                      +HUpm*HUm0C*(0)                                      +HUmp*HUm0C*(0)                                    +HUm0*HUm0C*(4977.908273555728)

    p=(1 + Pz)*np.absolute(App)**2 + (1 - Pz)*np.absolute(Amp)**2 + 2*np.real((Px - Py*1j)*App*np.conjugate(Amp))\
     +(1 + Pz)*np.absolute(Apm)**2 + (1 - Pz)*np.absolute(Amm)**2 + 2*np.real((Px - Py*1j)*Apm*np.conjugate(Amm))

    N = np.real(NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2)

    return p/N

#print('x')
#print(PDF(3.608286228594318, 1.796192503522510, -0.2, 0.0, np.pi, Px=0.5, Py=0.0, Pz=0.0, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))
#print(PDF(3.608286228594318, 1.796192503522510, 0, 0.2013579207900001, -np.pi/2, Px=0.5, Py=0.0, Pz=0.0, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))

#print('y')
#print(PDF(3.6082862285943182, 1.7961925035225106, 0, 1.3694384060000001, np.pi/2, Px=0.0, Py=0.5, Pz=0.0, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))
#print(PDF(3.6082862285943182, 1.7961925035225106, 0.2, np.pi/2, 0, Px=0.0, Py=0.5, Pz=0.0, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))

#print('z')
#print(PDF(3.608286228594318, 1.796192503522510, 0.9797958971120001, -1.0,  np.pi, Px=0.0, Py=0.0, Pz=0.5, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))
#print(PDF(3.608286228594318, 1.796192503522510, 0.9797958971120001, 2.4,  np.pi, Px=0.0, Py=0.0, Pz=0.5, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))

#print('0')
#print(PDF(3.608286228594318, 1.796192503522510, 0.5, 1.4, -1.52, Px=0.0, Py=0.0, Pz=0.0, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))
#print(PDF(3.608286228594318, 1.796192503522510, 0.8, -1.1, 2.1, Px=0.0, Py=0.0, Pz=0.0, HRpr=0.29, HRpi=0.04, HRmr=-0.16, HRmi=1.5, HSpr=-6.8, HSpi=3.1, HSmr=-13, HSmi=4.5, HUp0r=1, HUp0i=0, HUpmr=1.19, HUpmi=-1.03, HUmpr=-3.1, HUmpi=-3.3, HUm0r=-0.7, HUm0i=-4.2))
