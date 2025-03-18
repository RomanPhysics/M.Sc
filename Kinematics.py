#Roman Sultanov
import numpy as np
from numba import njit, prange

@njit
def beta(P): 
    # P = [E, px, py, pz]
    # β = pvec/E
    E = P[0]
    return np.array([P[1]/E, P[2]/E, P[3]/E])

@njit
def Lorentz(beta, P):
    # E' = γ (E - β·p)
    # p_i' = - γ β_i E + p_i + (γ - 1) β_i (β·p)/β²   for i=x,y,z
    beta2 = beta[0]*beta[0] + beta[1]*beta[1] + beta[2]*beta[2]
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    factor = (gamma - 1.0) / beta2
    bp = beta[0]*P[1] + beta[1]*P[2] + beta[2]*P[3]
    
    E_prime = gamma*(P[0] - bp)
    p1_prime = -gamma*beta[0]*P[0] + P[1] + factor * bp * beta[0]
    p2_prime = -gamma*beta[1]*P[0] + P[2] + factor * bp * beta[1] 
    p3_prime = -gamma*beta[2]*P[0] + P[3] + factor * bp * beta[2] 

    return np.array([E_prime, p1_prime, p2_prime, p3_prime])

@njit
def Rotation(α, β, γ, P):
    # Rotation defined using the Z-Y-Z convention
    cosα, sinα = np.cos(α), np.sin(α)
    cosβ, sinβ = np.cos(β), np.sin(β)
    cosγ, sinγ = np.cos(γ), np.sin(γ)

    R11 = cosα * cosβ * cosγ - sinα * sinγ
    R12 = -cosα * cosβ * sinγ - sinα * sinγ
    R13 = cosα * sinβ
    R21 = sinα * cosβ * cosγ + cosα * sinγ
    R22 = cosα * cosγ - sinα * cosβ * sinγ
    R23 = sinα * sinβ
    R31 = -sinβ * cosγ
    R32 = sinβ * sinγ
    R33 = cosβ

    px = P[1]
    py = P[2]
    pz = P[3]

    E_prime = P[0]
    p1_prime = R11*px + R12*py + R13*pz
    p2_prime = R21*px + R22*py + R23*pz
    p3_prime = R31*px + R32*py + R33*pz

    return np.array([E_prime, p1_prime, p2_prime, p3_prime])

@njit
def TwoBodyMomenta(mi, mf1, mf2):
    mi_squared = mi * mi
    mf1_plus_mf2 = mf1 + mf2
    mf1_minus_mf2 = mf1 - mf2

    term1 = mi_squared - mf1_plus_mf2 * mf1_plus_mf2
    term2 = mi_squared - mf1_minus_mf2 * mf1_minus_mf2
    return np.sqrt(term1 * term2) / (2 * mi)

@njit
def FinalStateMomenta(m2_12, m2_23, m0, m1, m2, m3):
    m2_13 = m0*m0 + m1*m1 + m2*m2 + m3*m3 - m2_12 - m2_23

    p1 = TwoBodyMomenta(m0, m1, np.sqrt(m2_23))
    p2 = TwoBodyMomenta(m0, m2, np.sqrt(m2_13))
    p3 = TwoBodyMomenta(m0, m3, np.sqrt(m2_12))

    p1_squared = p1 * p1
    inv_2p1 = 0.5 / p1
    cosθ2 = (p1_squared + p2*p2 - p3*p3)*inv_2p1/p2
    cosθ3 = (p1_squared + p3*p3 - p2*p2)*inv_2p1/p3

    sinθ2 = np.sqrt(1 - cosθ2*cosθ2)
    sinθ3 = np.sqrt(1 - cosθ3*cosθ3)

    #Fix 4-momenta with p3 oriented in z (quantisation axis) direction
    P1 = np.array([np.sqrt(p1_squared + m1 * m1), 0, 0, p1])
    P2 = np.array([np.sqrt(p2*p2 + m2*m2), p2*sinθ2, 0, -p2*cosθ2])
    P3 = np.array([np.sqrt(p3*p3 + m3*m3), -p3*sinθ3, 0, -p3*cosθ3])

    return P1, P2, P3

@njit
def FinalStateAngles(P1, P2, P3): #P=[E, px, py, pz]
    #R resonance
    PR = P1 + P2
    θR = np.arctan2(PR[1], PR[3])

    PR1 = Rotation(0, -θR, 0, Lorentz(beta(PR), P1)) #PR1 = L(PRz)R(0,-θR,-φR)P1 = R(0,-θR,-φR,)L(PR)P1
    θR1 = np.arctan2(PR1[1], PR1[3])

    bP1 = beta(P1)
    g1=1/np.sqrt(1 - bP1[0]**2 - bP1[1]**2 - bP1[2]**2)

    bPR = beta(PR)
    bPR1 = beta(PR1)
    gR=1/np.sqrt(1 - bPR[0]**2 - bPR[1]**2 - bPR[2]**2)
    gR1=1/np.sqrt(1 - bPR1[0]**2 - bPR1[1]**2 - bPR1[2]**2)
    aR=np.arccos((1 + g1 + gR + gR1)**2/((1 + g1)*(1 + gR)*(1 + gR1))-1)

    #S resonance
    PS = P1 + P3
    θS = np.arctan2(PS[1], PS[3])

    PS1 = Rotation(0, -θS, 0, Lorentz(beta(PS), P1)) #PS1 = L(PSz)R(0,-θS,-φS)P1 = R(0,-θS,-φS,)L(PS)P1
    θS1 = np.arctan2(PS1[1], PS1[3])

    bPS = beta(PS)
    bPS1 = beta(PS1)
    gS=1/np.sqrt(1 - bPS[0]**2 - bPS[1]**2 - bPS[2]**2)
    gS1=1/np.sqrt(1 - bPS1[0]**2 - bPS1[1]**2 - bPS1[2]**2)
    aS=np.arccos((1 + g1 + gS + gS1)**2/((1 + g1)*(1 + gS)*(1 + gS1))-1)

    #U resonance
    PbU2 = Lorentz(beta(P2+P3), P2) #PU2 = L(-PUz)R(0,-θ1,-φ1)P2 = R(0,-θU,-φU,)L(PU)P2
    θbU2 = np.arctan2(PbU2[1], PbU2[3])

    return θR, θR1, aR, θS, θS1, aS, θbU2
