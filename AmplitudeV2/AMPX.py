#Roman Sultanov
#Imports
from Form import *

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
def DecayrateDist(m2pK, m2Kπ, cosθp, φp, χ, Px, Py, Pz, HRpr, HRpi, HRmr, HRmi, HSpr, HSpi, HSmr, HSmi, HUp0r, HUp0i, HUpmr, HUpmi, HUmpr, HUmpi, HUm0r, HUm0i):
    #Es2 = (m2pK - mp ** 2 + mK ** 2) / (2 * np.sqrt(m2pK))
    #Es3 = (mΛ ** 2 - m2pK - mπ ** 2) / (2 * np.sqrt(m2pK))
    #sqrt_term1 = np.sqrt(Es2 ** 2 - mK ** 2)
    #sqrt_term2 = np.sqrt(Es3 ** 2 - mπ ** 2)
    #sq_term = (Es2 + Es3) ** 2
    #if m2pK <= (mp + mK) ** 2 or m2pK >= (mΛ - mπ) ** 2 or m2Kπ <= sq_term - (sqrt_term1 + sqrt_term2) ** 2 or m2Kπ >= sq_term - (sqrt_term1 - sqrt_term2) ** 2:
    #    return 0.0

    P1, P2, P3 = FinalStateMomenta(m2pK, m2Kπ, mΛ, mp, mK, mπ)
    θR, θR1, aR, θS, θS1, aS, θbU2 = FinalStateAngles(P1, P2, P3)

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

    NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2=HRp*HRpC*(7197.946564963944)                        +HRm*HRpC*(0)                                       +HSp*HRpC*(236.7670697929064+22.268528342668397j)    +HSm*HRpC*(-386.39396887382566+123.91358394887268j)  +HUp0*HRpC*(355.0942217927184-257.6806033247093j)    +HUpm*HRpC*(-132.86840918368028-143.02184952563496j) +HUmp*HRpC*(481.27856470507277-218.45935693926788j)+HUm0*HRpC*(-148.69998641805-67.52115627915572j)\
                                    +HRp*HRmC*(0)                                        +HRm*HRmC*(7197.946564963944)                       +HSp*HRmC*(386.39396887382566-123.91358394887268j)   +HSm*HRmC*(-236.7670697929064-22.267419968178384j)   +HUp0*HRmC*(148.69998641805+67.52115627915572j)      +HUpm*HRmC*(-481.27856470507277+218.45935693926788j) +HUmp*HRmC*(132.86840918368028+143.02184952563496j)+HUm0*HRmC*(-355.0942217927184+257.6806033247093j)\
                                    +HRp*HSpC*(236.7670697929064-22.268528342668397j)    +HRm*HSpC*(386.39396887382566+123.91358394887268j)  +HSp*HSpC*(1119.129511989152)                        +HSm*HSpC*(0)                                        +HUp0*HSpC*(-195.70335603078078+116.34693534292174j) +HUpm*HSpC*(22.108344526359897+78.32155932102214j)   +HUmp*HSpC*(331.82434772472743-24.256420298320936j)+HUm0*HSpC*(-27.344314365657663-43.57121126166499j)\
                                    +HRp*HSmC*(-386.39396887382566-123.91358394887268j)  +HRm*HSmC*(-236.7670697929064+22.267419968178384j)  +HSp*HSmC*(0)                                        +HSm*HSmC*(1119.129511989152)                        +HUp0*HSmC*(-27.344314365657663-43.57121126166499j)  +HUpm*HSmC*(331.82434772472743-24.256420298320936j)  +HUmp*HSmC*(22.108344526359897+78.32155932102214j) +HUm0*HSmC*(-195.70335603078078+116.34693534292174j)\
                                    +HRp*HUp0C*(355.0942217927184+257.6806033247093j)    +HRm*HUp0C*(148.69998641805-67.52115627915572j)     +HSp*HUp0C*(-195.70335603078078-116.34693534292174j) +HSm*HUp0C*(-27.344314365657663+43.57121126166499j)  +HUp0*HUp0C*(4977.908273555728)                      +HUpm*HUp0C*(0)                                      +HUmp*HUp0C*(0)                                    +HUm0*HUp0C*(0)\
                                    +HRp*HUpmC*(-132.86840918368028+143.02184952563496j) +HRm*HUpmC*(-481.27856470507277-218.45935693926788j)+HSp*HUpmC*(22.108344526359897-78.32155932102214j)   +HSm*HUpmC*(331.82434772472743+24.256420298320936j)  +HUp0*HUpmC*(0)                                      +HUpm*HUpmC*(4977.908273555728)                      +HUmp*HUpmC*(0)                                    +HUm0*HUpmC*(0)\
                                    +HRp*HUmpC*(481.27856470507277+218.45935693926788j)  +HRm*HUmpC*(132.86840918368028-143.02184952563496j) +HSp*HUmpC*(331.82434772472743+24.256420298320936j)  +HSm*HUmpC*(22.108344526359897-78.32155932102214j)   +HUp0*HUmpC*(0)                                      +HUpm*HUmpC*(0)                                      +HUmp*HUmpC*(4977.908273555728)                    +HUm0*HUmpC*(0)\
                                    +HRp*HUm0C*(-148.69998641805+67.52115627915572j)     +HRm*HUm0C*(-355.0942217927184-257.6806033247093j)  +HSp*HUm0C*(-27.344314365657663+43.57121126166499j)  +HSm*HUm0C*(-195.70335603078078-116.34693534292174j) +HUp0*HUm0C*(0)                                      +HUpm*HUm0C*(0)                                      +HUmp*HUm0C*(0)                                    +HUm0*HUm0C*(4977.908273555728)

    NAppAmpCNApmAmmC=HRp*HRpC*(0.02899747028916752+0.012147371007189128j)     +HRm*HRpC*(0)                                             +HSp*HRpC*(-0.0002715328069004792-0.0006808929852066067j)  +HSm*HRpC*(0.0026593676376462836-0.001964802944991977j)    +HUp0*HRpC*(0.006613151113113379-0.008638429681587895j)     +HUpm*HRpC*(0.00013638522592867932-0.012290396927453495j)    +HUmp*HRpC*(-0.002723344744272076+0.002093745822642243j)     +HUm0*HRpC*(-0.03023483064203033-0.0063931537449197825j)\
                    +HRp*HRmC*(0)                                             +HRm*HRmC*(0.033057943904826415+0.004144610375900447j)    +HSp*HRmC*(0.000010122597169878292-0.0014186230895719594j) +HSm*HRmC*(0.0017129555777339023+0.0009319889531004397j)   +HUp0*HRmC*(-0.0050881826086884045-0.00009333443528373607j) +HUpm*HRmC*(0.0038408634231398924+0.0028907088764852117j)    +HUmp*HRmC*(-0.0030723943384481224-0.0016541108889382603j)   +HUm0*HRmC*(0.008235840621921281+0.01961557940422942j)\
                    +HRp*HSpC*(-0.0029442620125103894+0.001983849494559106j)  +HRm*HSpC*(-0.0038735461930777624-0.001461662290159882j)  +HSp*HSpC*(0.004490367908629018-0.0018493006166730103j)    +HSm*HSpC*(0)                                              +HUp0*HSpC*(0.002237522521030022+0.0040203994910799215j)    +HUpm*HSpC*(0.0002942095370958616+0.006247493955445912j)     +HUmp*HSpC*(-0.0023073415454712976+0.001453120959972607j)    +HUm0*HSpC*(-0.004951564760809885-0.0006930719182145736j)\
                    +HRp*HSmC*(0.005149607634617716+0.006770863857285336j)    +HRm*HSmC*(-0.0009676327474298855-0.0008520510393864864j) +HSp*HSmC*(0)                                              +HSm*HSmC*(-0.006302619064550512-0.0026512164565702986j)   +HUp0*HSmC*(0.005062271032253302+0.0019761332400465704j)    +HUpm*HSmC*(-0.000908361402215572+0.0018840912461742692j)    +HUmp*HSmC*(0.0024396664332954373-0.006901188288003119j)     +HUm0*HSmC*(0.010288100346980245+0.0030739480686594665j)\
                    +HRp*HUp0C*(0.021632289489348464-0.00018247313918764085j) +HRm*HUp0C*(-0.0019659674330901513-0.003303807996375417j) +HSp*HUp0C*(-0.003430906120185042+0.002421828672855272j)   +HSm*HUp0C*(0.0074631926833489855+0.0030398956196586893j)  +HUp0*HUp0C*(-0.009218508137560704+0.013556651223174523j)   +HUpm*HUp0C*(0.01340042499259786-0.004186023008789228j)      +HUmp*HUp0C*(0)                                              +HUm0*HUp0C*(0)\
                    +HRp*HUpmC*(-0.003911160603597019-0.00848059811332659j)   +HRm*HUpmC*(-0.005564151359942884-0.0034977630683886325j) +HSp*HUpmC*(0.0007979550721321426-0.002577856936112897j)   +HSm*HUpmC*(-0.000347637898800122-0.0014638919702411983j)  +HUp0*HUpmC*(-0.010426748772663141+0.013809433381702122j)   +HUpm*HUpmC*(-0.001849862131221121-0.005784159136484755j)    +HUmp*HUpmC*(0)                                              +HUm0*HUpmC*(0)\
                    +HRp*HUmpC*(0.0019140708384736095-0.0017807411531090376j) +HRm*HUmpC*(0.0018400165805615178+0.0015939835574585303j) +HSp*HUmpC*(0.001216662579739549+0.0003874728918088142j)   +HSm*HUmpC*(-0.00761334734462846+0.002332071203796157j)    +HUp0*HUmpC*(0)                                             +HUpm*HUmpC*(0)                                              +HUmp*HUmpC*(0.010186772604771901+0.0026784329248031313j)    +HUm0*HUmpC*(-0.009205566867771554+0.015879614217715724j)\
                    +HRp*HUm0C*(-0.008453221722539503+0.007935915626373698j)  +HRm*HUm0C*(0.015337862131958884-0.006750340396010849j)   +HSp*HUm0C*(0.0019055271458750642+0.0006565586882572647j)  +HSm*HUm0C*(0.008950846798553548+0.002667355433701822j)    +HUp0*HUm0C*(0)                                             +HUpm*HUm0C*(0)                                              +HUmp*HUm0C*(-0.013661025289341796-0.0013317573018025157j)   +HUm0*HUm0C*(0.01258493019998163-0.017181448353783296j)


    p=(1 + Pz)*np.absolute(App)**2 + (1 - Pz)*np.absolute(Amp)**2 + 2*np.real((Px - Py*1j)*App*np.conjugate(Amp))\
     +(1 + Pz)*np.absolute(Apm)**2 + (1 - Pz)*np.absolute(Amm)**2 + 2*np.real((Px - Py*1j)*Apm*np.conjugate(Amm))

    N = np.real(NAbsApp2NAbsApm2NAbsAmp2NAbsAmm2 + 2*(Px - Py*1j)*NAppAmpCNApmAmmC)

    return p/N