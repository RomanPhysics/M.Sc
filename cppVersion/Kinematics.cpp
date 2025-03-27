#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <tuple>
#include <iomanip>


inline Eigen::Vector3d beta(const Eigen::Vector4d& P) {
    double E = P[0];
    return P.segment<3>(1) / E;
}

inline Eigen::Vector4d Lorentz(const Eigen::Vector3d& beta, const Eigen::Vector4d& P) {
    const double beta2 = beta.squaredNorm();
    const double gamma = 1.0 / std::sqrt(1.0 - beta2);
    const double factor = (gamma - 1.0) / beta2;
    const Eigen::Vector3d p = P.segment<3>(1);
    double bp = beta.dot(p);
    
    const double E_prime = gamma * (P[0] - bp);
    Eigen::Vector3d p_prime = -gamma * beta * P[0] + p + factor * bp * beta;

    Eigen::Vector4d result;
    result[0] = E_prime;
    result.template segment<3>(1) = p_prime;
    return result;
}

inline Eigen::Vector4d Rotation(double α, double β, double γ, const Eigen::Vector4d& P) {
    const double cosα = std::cos(α);
    const double sinα = std::sin(α);
    const double cosβ = std::cos(β);
    const double sinβ = std::sin(β);
    const double cosγ = std::cos(γ);
    const double sinγ = std::sin(γ);

    const double R11 = cosα * cosβ * cosγ - sinα * sinγ;
    const double R12 = -cosα * cosβ * sinγ - sinα * sinγ;
    const double R13 = cosα * sinβ;
    const double R21 = sinα * cosβ * cosγ + cosα * sinγ;
    const double R22 = cosα * cosγ - sinα * cosβ * sinγ;
    const double R23 = sinα * sinβ;
    const double R31 = -sinβ * cosγ;
    const double R32 = sinβ * sinγ;
    const double R33 = cosβ;

    Eigen::Vector4d result;
    const double px = P[1], py = P[2], pz = P[3];

    result[0] = P[0];
    result[1] = R11 * px + R12 * py + R13 * pz;
    result[2] = R21 * px + R22 * py + R23 * pz;
    result[3] = R31 * px + R32 * py + R33 * pz;

    return result;
}

inline double TwoBodyMomenta(double mi, double mf1, double mf2) {
    const double mi_squared = mi * mi;
    const double mf1_plus_mf2 = mf1 + mf2;
    const double mf1_minus_mf2 = mf1 - mf2;

    const double term1 = mi_squared - mf1_plus_mf2 * mf1_plus_mf2;
    const double term2 = mi_squared - mf1_minus_mf2 * mf1_minus_mf2;
    return std::sqrt(term1 * term2) / (2.0 * mi);
}

inline Eigen::Matrix<double, 4, 3> FinalStateMomenta(double m2_12, double m2_23, double m0, double m1, double m2, double m3) {
    const double m2_13 = m0*m0 + m1*m1 + m2*m2 + m3*m3 - m2_12 - m2_23;

    const double p1 = TwoBodyMomenta(m0, m1, std::sqrt(m2_23));
    const double p2 = TwoBodyMomenta(m0, m2, std::sqrt(m2_13));
    const double p3 = TwoBodyMomenta(m0, m3, std::sqrt(m2_12));

    const double p1_squared = p1 * p1;
    const double inv_2p1 = 0.5 / p1;
    const double cosθ2 = (p1_squared + p2*p2 - p3*p3) * inv_2p1 / p2;
    const double cosθ3 = (p1_squared + p3*p3 - p2*p2) * inv_2p1 / p3;

    const double sinθ2 = std::sqrt(1.0 - cosθ2*cosθ2);
    const double sinθ3 = std::sqrt(1.0 - cosθ3*cosθ3);

    // Construct the 4-vectors (with the third spatial component = 0)
    Eigen::Vector4d P1, P2, P3;
    P1 << std::sqrt(p1_squared + m1*m1), 0.0, 0.0, p1;
    P2 << std::sqrt(p2*p2 + m2*m2), p2 * sinθ2, 0.0, -p2 * cosθ2;
    P3 << std::sqrt(p3*p3 + m3*m3), -p3 * sinθ3, 0.0, -p3 * cosθ3;

    Eigen::Matrix<double, 4, 3> finalMomenta;
    finalMomenta.col(0) = P1;
    finalMomenta.col(1) = P2;
    finalMomenta.col(2) = P3;

    return finalMomenta;
}

inline Eigen::Matrix<double, 7, 1> FinalStateAngles(const Eigen::Matrix<double, 4, 3>& finalMomenta) {
    const Eigen::Vector4d P1 = finalMomenta.col(0);
    const Eigen::Vector4d P2 = finalMomenta.col(1);
    const Eigen::Vector4d P3 = finalMomenta.col(2);


    //R resonance
    const Eigen::Vector4d PR = P1 + P2;
    const double θR = std::atan2(PR[1], PR[3]);

    const Eigen::Vector3d bPR = beta(PR);
    const Eigen::Vector4d PR1 = Rotation(0.0, -θR, 0.0, Lorentz(beta(PR), P1)); //PR1 = L(PRz)R(0,-θR,-φR)P1 = R(0,-θR,-φR,)L(PR)P1
    const double θR1 = std::atan2(PR1[1], PR1[3]);

    const Eigen::Vector3d bP1 = beta(P1);
    const Eigen::Vector3d bPR1 = beta(PR1);
    const double g1 = 1.0 / std::sqrt(1.0 - bP1.squaredNorm());
    const double gR = 1.0 / std::sqrt(1.0 - bPR.squaredNorm());
    const double gR1 = 1.0 / std::sqrt(1.0 - bPR1.squaredNorm());
    const double aR = std::acos(std::pow(1.0 + g1 + gR + gR1, 2) / ((1.0 + g1) * (1.0 + gR) * (1.0 + gR1)) - 1.0 );


    //S resonance
    const Eigen::Vector4d PS = P1 + P3;
    const double θS = std::atan2(PS[1], PS[3]);

    const Eigen::Vector3d bPS = beta(PS);
    const Eigen::Vector4d PS1 = Rotation(0.0, -θS, 0.0, Lorentz(beta(PS), P1));
    const double θS1 = std::atan2(PS1[1], PS1[3]);

    const Eigen::Vector3d bPS1 = beta(PS1);
    const double gS = 1.0 / std::sqrt(1.0 - bPS.squaredNorm());
    const double gS1 = 1.0 / std::sqrt(1.0 - bPS1.squaredNorm());
    const double aS = std::acos(std::pow(1.0 + g1 + gS + gS1, 2) / ((1.0 + g1) * (1.0 + gS) * (1.0 + gS1)) - 1.0 );


    //U resonance
    const Eigen::Vector4d PbU2 = Lorentz(beta(P2 + P3), P2);
    const double θbU2 = std::atan2(PbU2[1], PbU2[3]);

    Eigen::Matrix<double, 7, 1> angles;
    angles << θR, θR1, aR, θS, θS1, aS, θbU2;
    return angles;
}



int main() {
    double m0 = 2.2865;  // lambda_c^+ mass
    double m1 = 0.9383;  // proton mass
    double m2 = 0.4937;  // kaon- mass
    double m3 = 0.1396;  // pion+ mass
    double m2_12 = 2.5;
    double m2_23 = 1.6;

    Eigen::Matrix<double, 4, 3> finalMomenta;
    Eigen::Matrix<double, 7, 1> angles;

    auto start = std::chrono::high_resolution_clock::now();
    finalMomenta = FinalStateMomenta(m2_12, m2_23, m0, m1, m2, m3);
    angles = FinalStateAngles(finalMomenta);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Elapsed time: " << diff.count() << " seconds" << std::endl;


    std::cout << std::setprecision(17) << "P1: " << finalMomenta.col(0).transpose() << std::endl;
    std::cout << std::setprecision(17) << "P2: " << finalMomenta.col(1).transpose() << std::endl;
    std::cout << std::setprecision(17) << "P3: " << finalMomenta.col(2).transpose() << std::endl;
    std::cout << std::setprecision(17) << "Angles: " << angles.transpose() << std::endl;

    return 0;

}
