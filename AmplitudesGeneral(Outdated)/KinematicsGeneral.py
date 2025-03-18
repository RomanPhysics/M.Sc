#Roman Sultanov
import numpy as np
from numba import njit, prange

@njit
def betaCM(Psum): #Psum=P1+P2+...
    return Psum[1:]/Psum[0]#βvec = pvecsum/Esum

@njit
def Lorentz(beta, P): #Λ(βvec)P
    beta2 = np.dot(beta, beta)
    gamma = 1 / np.sqrt(1 - beta2)
    gamma_minus_one = gamma - 1

    L = np.empty((4, 4), dtype=P.dtype)
    L[0, 0] = gamma
    L[0, 1:] = L[1:, 0] = -gamma * beta
    L[1:, 1:] = gamma_minus_one * np.outer(beta, beta) / beta2
    L[1:, 1:] += np.eye(3)

    return np.dot(L, P)

@njit
def Rotation(α, β, γ, P):
    cosα, sinα = np.cos(α), np.sin(α)
    cosβ, sinβ = np.cos(β), np.sin(β)
    cosγ, sinγ = np.cos(γ), np.sin(γ)

    R = np.empty((4, 4), dtype=P.dtype)
    R[0, 0] = 1
    R[0, 1:] = R[1:, 0] = 0

    R[1, 1] = cosα * cosβ * cosγ - sinα * sinγ
    R[1, 2] = -cosα * cosβ * sinγ - sinα * sinγ
    R[1, 3] = cosα * sinβ
    R[2, 1] = sinα * cosβ * cosγ + cosα * sinγ
    R[2, 2] = cosα * cosγ - sinα * cosβ * sinγ
    R[2, 3] = sinα * sinβ
    R[3, 1] = -sinβ * cosγ
    R[3, 2] = sinβ * sinγ
    R[3, 3] = cosβ

    return np.dot(R, P)

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

    #Magnitude of the momenta
    p1 = TwoBodyMomenta(m0, m1, np.sqrt(m2_23))
    p2 = TwoBodyMomenta(m0, m2, np.sqrt(m2_13))
    p3 = TwoBodyMomenta(m0, m3, np.sqrt(m2_12))

    p1_squared = p1 * p1
    inv_2p1 = 0.5 / p1
    cosθ2 = (p1_squared + p2 * p2 - p3 * p3) * inv_2p1 / p2
    cosθ3 = (p1_squared + p3 * p3 - p2 * p2) * inv_2p1 / p3

    sinθ2 = np.sqrt(1 - cosθ2 * cosθ2)
    sinθ3 = np.sqrt(1 - cosθ3 * cosθ3)

    #Fix 4-momenta with p3 oriented in z (quantisation axis) direction
    P1 = np.array([np.sqrt(p1_squared + m1 * m1), 0, 0, p1])
    P2 = np.array([np.sqrt(p2 * p2 + m2 * m2), p2 * sinθ2, 0, -p2 * cosθ2])
    P3 = np.array([np.sqrt(p3 * p3 + m3 * m3), -p3 * sinθ3, 0, -p3 * cosθ3])

    #Rotate 4-momenta to correct directions (Z-Y-Z convention)
    return P1, P2, P3

@njit
def FinalStateAngles(P1, P2, P3): #P=[E, px, py, pz]
    #R resonance
    PR = P1 + P2
    θR = np.arctan2(PR[1], PR[3])

    PR1 = Rotation(0, -θR, 0, Lorentz(betaCM(PR), P1)) #PR1 = L(PRz)R(0,-θR,-φR)P1 = R(0,-θR,-φR,)L(PR)P1
    θR1 = np.arctan2(PR1[1], PR1[3])

    g1=1/np.sqrt(1-np.linalg.norm(betaCM(P1))**2)

    gR=1/np.sqrt(1-np.linalg.norm(betaCM(PR))**2)
    gR1=1/np.sqrt(1-np.linalg.norm(betaCM(PR1))**2)
    aR=np.arccos((1+g1+gR+gR1)**2/((1+g1)*(1+gR)*(1+gR1))-1)

    #S resonance
    PS = P1 + P3
    θS = np.arctan2(PS[1], PS[3])

    PS1 = Rotation(0, -θS, 0, Lorentz(betaCM(PS), P1)) #PS1 = L(PSz)R(0,-θS,-φS)P1 = R(0,-θS,-φS,)L(PS)P1
    θS1 = np.arctan2(PS1[1], PS1[3])

    gS=1/np.sqrt(1-np.linalg.norm(betaCM(PS))**2)
    gS1=1/np.sqrt(1-np.linalg.norm(betaCM(PS1))**2)
    aS=np.arccos((1+g1+gS+gS1)**2/((1+g1)*(1+gS)*(1+gS1))-1)

    #U resonance
    PbU2 = Lorentz(betaCM(P2+P3), P2) #PU2 = L(-PUz)R(0,-θ1,-φ1)P2 = R(0,-θU,-φU,)L(PU)P2
    θbU2 = np.arctan2(PbU2[1], PbU2[3])

    return θR, θR1, aR, θS, θS1, aS, θbU2
