#include <iostream>
#include <cmath>
#include <functional>

// implement adaptive Gaussian quadrature as in Mathematica
double adaptiveGaussQuadrature(std::function<double(double)> func, double a, double b, double tol = 1e-6) {
    double x1 = (a + b) / 2.0;
    double x2 = (b - a) / 2.0;
    double x[5] = {0, -0.538469, 0.538469, -0.90618, 0.90618};
    double w[5] = {128.0/225.0, 0.47862867049936647, 0.47862867049936647, 0.23692688505618908, 0.23692688505618908};
    double sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += w[i] * func(x1 + x[i] * x2);
    }
    double integral = x2 * sum;
    return integral;
}

// Simple bisection method to find the root of F(z_s)=0 in [low, high]
double bisection(std::function<double(double)> F_func, double low, double high, double tol = 1e-6, int max_iter = 1000) {
    double mid;
    for (int i = 0; i < max_iter; ++i) {
        mid = (low + high) / 2.0;
        double F_mid = F_func(mid);
        if (std::fabs(F_mid) < tol)
            return mid;
        if (F_func(low) * F_mid < 0)
            high = mid;
        else
            low = mid;
    }
    return mid; // return the best estimate if tolerance not met
}

// Define F(z_s) = integral from z_E to z_s of f(z') dz' - 1
double F(double z_s, double z_E, double E) {
    auto integrand = [=](double z) {
        return (1+z)*(1+z)/40*(E/20) * (-3.0e5/(70*sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7)*(1+z)));
    };
    double integral = adaptiveGaussQuadrature(integrand, z_E, z_s); // adjust subdivisions if needed
    return integral-1.0;
}

double calc_z(double E, double z_E, double low, double high) {
    // Create a lambda for F(z_s) with fixed z_E and function f
    auto F_func = [=](double z_s) { return F(z_s, z_E, E); };
    double z_s = bisection(F_func, low, high);
    return z_s;
}

// signum function
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}