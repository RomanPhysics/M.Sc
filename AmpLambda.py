#Roman Sultanov
#Imports
from Kinematics import *

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
def PDF(m2pK, m2Kπ, cosθp, φp, χ, Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUp0r, HUp0i, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i):
    #Es2 = (m2pK - mp ** 2 + mK ** 2) / (2 * np.sqrt(m2pK))
    #Es3 = (mΛ ** 2 - m2pK - mπ ** 2) / (2 * np.sqrt(m2pK))
    #sqrt_term1 = np.sqrt(Es2 ** 2 - mK ** 2)
    #sqrt_term2 = np.sqrt(Es3 ** 2 - mπ ** 2)
    #sq_term = (Es2 + Es3) ** 2
    #if m2pK <= (mp + mK) ** 2 or m2pK >= (mΛ - mπ) ** 2 or m2Kπ <= sq_term - (sqrt_term1 + sqrt_term2) ** 2 or m2Kπ >= sq_term - (sqrt_term1 - sqrt_term2) ** 2:
    #    return 0.0

    P1, P2, P3 = FinalStateMomenta(m2pK, m2Kπ, mΛ, mp, mK, mπ)
    θR, θR1, aR, θS, θS1, aS, θbU2 = FinalStateAngles(P1, P2, P3)

    HRp=HRpr   + HRpi*1j
    HRm=HRmr   + HRmi*1j
    HSp=HSpr   + HSpi*1j
    HSm=HSmr   + HSmi*1j
    HUp0=HUp0r + HUp0i*1j
    HUpm=HUpmr + HUpmi*1j
    HUmp=HUmpr + HUmpi*1j
    HUm0=HUm0r + HUm0i*1j

    HRpC=HRpr   - HRpi*1j
    HRmC=HRmr   - HRmi*1j
    HSpC=HSpr   - HSpi*1j
    HSmC=HSmr   - HSmi*1j
    HUp0C=HUp0r - HUp0i*1j
    HUpmC=HUpmr - HUpmi*1j
    HUmpC=HUmpr - HUmpi*1j
    HUm0C=HUm0r - HUm0i*1j

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

    App = HRp*ppC1 + HRm*ppC2 +HSp*ppC3 + HSm*ppC4 + HUp0*ppC5 + HUpm*ppC6 + HUmp*ppC7 + HUm0*ppC8

    pmC1=(WdAR_12mp*WdR_12pp*WdR1_32pp*BWR-WdAR_12mm*WdR_12pp*WdR1_32pm*BWR)*cWDE_12pp+(WdAR_12mp*WdR_12mp*WdR1_32pp*BWR-WdAR_12mm*WdR_12mp*WdR1_32pm*BWR)*cWDE_12pm
    pmC2=(WdAR_12mp*WdR_12pm*WdR1_32mp*BWR-WdAR_12mm*WdR_12pm*WdR1_32mm*BWR)*cWDE_12pp+(WdAR_12mp*WdR_12mm*WdR1_32mp*BWR-WdAR_12mm*WdR_12mm*WdR1_32mm*BWR)*cWDE_12pm
    pmC3=(WdAS_12mp*WdS_12pp*WdS1_32pp*BWS+WdAS_12mm*WdS_12pp*WdS1_32pm*BWS)*cWDE_12pp+(WdAS_12mp*WdS_12mp*WdS1_32pp*BWS+WdAS_12mm*WdS_12mp*WdS1_32pm*BWS)*cWDE_12pm
    pmC4=(WdAS_12mp*WdS_12pm*WdS1_32mp*BWS+WdAS_12mm*WdS_12pm*WdS1_32mm*BWS)*cWDE_12pp+(WdAS_12mp*WdS_12mm*WdS1_32mp*BWS+WdAS_12mm*WdS_12mm*WdS1_32mm*BWS)*cWDE_12pm
    pmC5=0
    pmC6=0
    pmC7=WdbU2_1p0*cWDE_12pp*BWU
    pmC8=WdbU2_100*cWDE_12pm*BWU

    Apm = HRp*pmC1 + HRm*pmC2 +HSp*pmC3 + HSm*pmC4 + HUp0*pmC5 + HUpm*pmC6 + HUmp*pmC7 + HUm0*pmC8

    mpC1=(WdAR_12pp*WdR_12pp*WdR1_32pp*BWR-WdAR_12pm*WdR_12pp*WdR1_32pm*BWR)*cWDE_12mp+(WdAR_12pp*WdR_12mp*WdR1_32pp*BWR-WdAR_12pm*WdR_12mp*WdR1_32pm*BWR)*cWDE_12mm
    mpC2=(WdAR_12pp*WdR_12pm*WdR1_32mp*BWR-WdAR_12pm*WdR_12pm*WdR1_32mm*BWR)*cWDE_12mp+(WdAR_12pp*WdR_12mm*WdR1_32mp*BWR-WdAR_12pm*WdR_12mm*WdR1_32mm*BWR)*cWDE_12mm
    mpC3=(WdAS_12pp*WdS_12pp*WdS1_32pp*BWS+WdAS_12pm*WdS_12pp*WdS1_32pm*BWS)*cWDE_12mp+(WdAS_12pp*WdS_12mp*WdS1_32pp*BWS+WdAS_12pm*WdS_12mp*WdS1_32pm*BWS)*cWDE_12mm
    mpC4=(WdAS_12pp*WdS_12pm*WdS1_32mp*BWS+WdAS_12pm*WdS_12pm*WdS1_32mm*BWS)*cWDE_12mp+(WdAS_12pp*WdS_12mm*WdS1_32mp*BWS+WdAS_12pm*WdS_12mm*WdS1_32mm*BWS)*cWDE_12mm
    mpC5=WdbU2_100*cWDE_12mp*BWU
    mpC6=WdbU2_1m0*cWDE_12mm*BWU
    mpC7=0
    mpC8=0

    Amp = HRp*mpC1 + HRm*mpC2 +HSp*mpC3 + HSm*mpC4 + HUp0*mpC5 + HUpm*mpC6 + HUmp*mpC7 + HUm0*mpC8

    mmC1=(WdAR_12mp*WdR_12pp*WdR1_32pp*BWR-WdAR_12mm*WdR_12pp*WdR1_32pm*BWR)*cWDE_12mp+(WdAR_12mp*WdR_12mp*WdR1_32pp*BWR-WdAR_12mm*WdR_12mp*WdR1_32pm*BWR)*cWDE_12mm
    mmC2=(WdAR_12mp*WdR_12pm*WdR1_32mp*BWR-WdAR_12mm*WdR_12pm*WdR1_32mm*BWR)*cWDE_12mp+(WdAR_12mp*WdR_12mm*WdR1_32mp*BWR-WdAR_12mm*WdR_12mm*WdR1_32mm*BWR)*cWDE_12mm
    mmC3=(WdAS_12mp*WdS_12pp*WdS1_32pp*BWS+WdAS_12mm*WdS_12pp*WdS1_32pm*BWS)*cWDE_12mp+(WdAS_12mp*WdS_12mp*WdS1_32pp*BWS+WdAS_12mm*WdS_12mp*WdS1_32pm*BWS)*cWDE_12mm
    mmC4=(WdAS_12mp*WdS_12pm*WdS1_32mp*BWS+WdAS_12mm*WdS_12pm*WdS1_32mm*BWS)*cWDE_12mp+(WdAS_12mp*WdS_12mm*WdS1_32mp*BWS+WdAS_12mm*WdS_12mm*WdS1_32mm*BWS)*cWDE_12mm
    mmC5=0
    mmC6=0
    mmC7=WdbU2_1p0*cWDE_12mp*BWU
    mmC8=WdbU2_100*cWDE_12mm*BWU

    Amm = HRp*mmC1 + HRm*mmC2 +HSp*mmC3 + HSm*mmC4 + HUp0*mmC5 + HUpm*mmC6 + HUmp*mmC7 + HUm0*mmC8

    NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2=HRp*HRpC*(5311.814872048573)                        +HRm*HRpC*(0)                                       +HSp*HRpC*(204.6937609247803-6.007421260172587j)     +HSm*HRpC*(-287.2560475344617+149.64058433328609j)   +HUp0*HRpC*(189.99171355418065-266.8169844880696j)   +HUpm*HRpC*(-172.1689365807526-91.80621696804803j) +HUmp*HRpC*(295.5023133321682-314.48754754001925j) +HUm0*HRpC*(-157.61475492541712-25.851853170771335j)\
                                    +HRp*HRmC*(0)                                        +HRm*HRmC*(5311.814872048573)                       +HSp*HRmC*(287.2560475344617-149.64058433328609j)    +HSm*HRmC*(-204.6937609247803+6.007421260172587j)    +HUp0*HRmC*(157.61475492541712+25.851853170771335j)  +HUpm*HRmC*(-295.5023133321682+314.48754754001925j)+HUmp*HRmC*(172.1689365807526+91.80621696804803j)  +HUm0*HRmC*(-189.99171355418065+266.8169844880696j)\
                                    +HRp*HSpC*(204.6937609247803+6.007421260172587j)     +HRm*HSpC*(287.2560475344617+149.64058433328609j)   +HSp*HSpC*(823.9195265936702)                        +HSm*HSpC*(0)                                        +HUp0*HSpC*(-136.65038380072326+134.79009858281137j) +HUpm*HSpC*(38.64034900132792+74.67096150196075j)  +HUmp*HSpC*(258.85864285874237-57.46888540504888j) +HUm0*HSpC*(-25.181691922845722-40.746441452622705j)\
                                    +HRp*HSmC*(-287.2560475344617-149.64058433328609j)   +HRm*HSmC*(-204.6937609247803-6.007421260172587j)   +HSp*HSmC*(0)                                        +HSm*HSmC*(823.9195265936702)                        +HUp0*HSmC*(-25.181691922845722-40.746441452622705j) +HUpm*HSmC*(258.85864285874237-57.46888540504888j) +HUmp*HSmC*(38.64034900132792+74.67096150196075j)  +HUm0*HSmC*(-136.65038380072326+134.79009858281137j)\
                                    +HRp*HUp0C*(189.99171355418065+266.8169844880696j)   +HRm*HUp0C*(157.61475492541712-25.851853170771335j) +HSp*HUp0C*(-136.65038380072326-134.79009858281137j) +HSm*HUp0C*(-25.181691922845722+40.746441452622705j) +HUp0*HUp0C*(3735.0995605146118)                     +HUpm*HUp0C*(0)                                    +HUmp*HUp0C*(0)                                    +HUm0*HUp0C*(0)\
                                    +HRp*HUpmC*(-172.1689365807526+91.80621696804803j)   +HRm*HUpmC*(-295.5023133321682-314.48754754001925j) +HSp*HUpmC*(38.64034900132792-74.67096150196075j)    +HSm*HUpmC*(258.85864285874237+57.46888540504888j)   +HUp0*HUpmC*(0)                                      +HUpm*HUpmC*(3735.0995605146118)                   +HUmp*HUpmC*(0)                                    +HUm0*HUpmC*(0)\
                                    +HRp*HUmpC*(295.5023133321682+314.48754754001925j)   +HRm*HUmpC*(172.1689365807526-91.80621696804803j)   +HSp*HUmpC*(258.85864285874237+57.46888540504888j)   +HSm*HUmpC*(38.64034900132792-74.67096150196075j)    +HUp0*HUmpC*(0)                                      +HUpm*HUmpC*(0)                                    +HUmp*HUmpC*(3735.0995605146118)                   +HUm0*HUmpC*(0)\
                                    +HRp*HUm0C*(-157.61475492541712+25.851853170771335j) +HRm*HUm0C*(-189.99171355418065-266.8169844880696j) +HSp*HUm0C*(-25.181691922845722+40.746441452622705j) +HSm*HUm0C*(-136.65038380072326-134.79009858281137j) +HUp0*HUm0C*(0)                                      +HUpm*HUm0C*(0)                                    +HUmp*HUm0C*(0)                                    +HUm0*HUm0C*(3735.0995605146118)

    p=(1 + Pz)*np.absolute(App)**2 + (1 - Pz)*np.absolute(Amp)**2 + 2*np.real((Px - Py*1j)*App*np.conjugate(Amp))\
     +(1 + Pz)*np.absolute(Apm)**2 + (1 - Pz)*np.absolute(Amm)**2 + 2*np.real((Px - Py*1j)*Apm*np.conjugate(Amm))

    N = np.real(NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2)

    return p/N