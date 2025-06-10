#include "helper.h"
#include <cmath>

double adaptiveGaussQuadrature(std::function<double(double)> func,
                               double a, double b,
                               double tol) {
    double x1 = (a + b) / 2.0;
    double x2 = (b - a) / 2.0;
    double x[5] = {0, -0.538469, 0.538469, -0.90618, 0.90618};
    double w[5] = {128.0/225.0, 0.47862867049936647, 0.47862867049936647,
                   0.23692688505618908, 0.23692688505618908};
    double sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += w[i] * func(x1 + x[i] * x2);
    }
    return x2 * sum;
}

double bisection(std::function<double(double)> F_func,
                 double low, double high,
                 double tol, int max_iter) {
    double mid = low;
    for (int i = 0; i < max_iter; ++i) {
        mid = (low + high) / 2.0;
        double fm = F_func(mid);
        if (std::fabs(fm) < tol) return mid;
        if (F_func(low) * fm < 0)
            high = mid;
        else
            low = mid;
    }
    return mid;
}

double F(double z_s, double z_E, double E) {
    auto integrand = [=](double z) {
        return (1+z)*(1+z)/40*(E/20)
             * (-3.0e5/(70*std::sqrt(0.3*(1+z)*(1+z)*(1+z)+0.7)
             * (1+z)));
    };
    double integral = adaptiveGaussQuadrature(integrand, z_E, z_s);
    return integral - 1.0;
}

// calculate scattering redshift
double calc_z(double E, double z_E, double low, double high) {
    auto F_func = [=](double zs) { return F(zs, z_E, E); };
    return bisection(F_func, low, high);
}

// axis-angle rotation matrix
Mat3 rotation_matrix(Vec3 const& ax, double angle) {
    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1 - c;
    double x = ax.x, y = ax.y, z = ax.z;
    double x2 = x*x, y2 = y*y, z2 = z*z;
    double xy = x*y, xz = x*z, yz = y*z;
    return Mat3{{{t*x2 + c,   t*xy - s*z, t*xz + s*y},
                {t*xy + s*z, t*y2 + c,   t*yz - s*x},
                {t*xz - s*y, t*yz + s*x, t*z2 + c}}};
}

// rotation from vector a to b
Mat3 RotA2B(Vec3 const& a, Vec3 const& b) {
    Vec3 axis = a.cross(b);
    double c = a.dot(b);
    double s = axis.norm();
    if (s < 1e-12) return c * id_mat();
    double t = 1 - c;
    Mat3 vx = Mat3{{{0, -axis.z, axis.y},
                    {axis.z, 0, -axis.x},
                    {-axis.y, axis.x, 0}}};
    return id_mat() + vx + vx*vx * (t/(s*s));
}

// convert jet-frame to galactic-frame
Mat3 jet2gal(Vec3 const& LOS_dir, Vec3 const& src_dir) {
    Vec3 zprime{0,0,1};
    Mat3 R1 = RotA2B(LOS_dir, zprime);
    Mat3 R2 = RotA2B(zprime, -src_dir);
    return R2 * R1;
}