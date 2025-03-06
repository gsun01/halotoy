#include <iostream>
#include <cmath>
#include <functional>

double f(double z) {
    // integrand
    return 3e5*(1+z) / (70*sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7));
}

double simpsonsRule(std::function<double(double)> func, double a, double b, int n) {
    if (n % 2 != 0) n++; // ensure n is even
    double h = (b - a) / n;
    double s = func(a) + func(b);
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        s += (i % 2 == 0 ? 2.0 : 4.0) * func(x);
    }
    return s * h / 3.0;
}

// Define F(z_s) = integral from z_E to z_s of f(z') dz' - 1
double F(double z_s, double z_E, double E, std::function<double(double)> func) {
    double integral = simpsonsRule(func, z_E, z_s, 1000); // adjust subdivisions if needed
    return 40/(E/20) + integral;
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

double calc_z(double E, double z_E, double low, double high) {
    // Create a lambda for F(z_s) with fixed z_E and function f
    auto F_func = [=](double z_s) { return F(z_s, z_E, E, f); };

    double z_s = bisection(F_func, low, high);
    return z_s;
}