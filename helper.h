#pragma once

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

// 3D vector and matrix operations
struct Vec3 { double x,y,z;
  Vec3 operator+(Vec3 const& o) const {return {x+o.x,y+o.y,z+o.z};}
  Vec3 operator-(Vec3 const& o) const {return {x-o.x,y-o.y,z-o.z};}
  Vec3 operator-() const { return { -x, -y, -z }; }
  Vec3 operator*(double s) const {return {x*s,y*s,z*s};}
  Vec3 cross(Vec3 const& o) const {
    return {y*o.z - z*o.y,
            z*o.x - x*o.z,
            x*o.y - y*o.x};
  }
  double dot(Vec3 const& o) const {
    return x*o.x + y*o.y + z*o.z;
  }
  double norm() const { return std::sqrt(x*x + y*y + z*z); }
  Vec3 normalized() const {
    double n = norm();
    if (n <= 1e-10) return {0, 0, 0}; // avoid division by zero
    return {x / n, y / n, z / n};
  }
};
static Vec3 operator*(double s, Vec3 const& v) {
  return { v.x * s, v.y * s, v.z * s };
}


struct Mat3 {
  double m[3][3];
  Vec3 operator*(Vec3 const& v) const {
    return {m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z};
  }
  Mat3 operator*(Mat3 const& o) const {
    Mat3 res = {};
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        for (int k=0; k<3; ++k)
          res.m[i][j] += m[i][k] * o.m[k][j];
    return res;
  }
  Mat3 operator+(Mat3 const& o) const {
    Mat3 res = {};
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        res.m[i][j] = m[i][j] + o.m[i][j];
    return res;
  }
  Mat3 operator-(Mat3 const& o) const {
    Mat3 res = {};
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        res.m[i][j] = m[i][j] - o.m[i][j];
    return res;
  }
  Mat3 operator-() const {
    Mat3 res = {};
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        res.m[i][j] = -m[i][j];
    return res;
  }
  Mat3 operator*(double s) const {
    Mat3 res = {};
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        res.m[i][j] = m[i][j] * s;
    return res;
  }
};
static Mat3 operator*(double s, Mat3 const& mat) {
    Mat3 res = {};
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        res.m[i][j] = mat.m[i][j] * s;
    return res;
}

static Mat3 id_mat() {
    return Mat3{{{1, 0, 0},
                 {0, 1, 0},
                 {0, 0, 1}}};
}

Mat3 RotA2B(Vec3 const& a, Vec3 const& b) {
    // returns matrix R such that Ra = b
    // assumes a and b are normalized 3D vectors
    Vec3 axis = a.cross(b);
    double ax_norm = axis.norm();
    if (ax_norm < 1e-10) {
        return id_mat();
    }
    double c = a.dot(b);
    double s = ax_norm;
    double t = 1.0 - c;

    Mat3 vx = Mat3 {
        {{0, -axis.z, axis.y},
        {axis.z, 0, -axis.x},
        {-axis.y, axis.x, 0}}
    };
    Mat3 R = id_mat() + vx + vx*vx * (t/(s*s));
    return R;
}

Vec3 sph2cart(double theta, double phi) {
  double sin_theta = std::sin(theta);
    return {sin_theta * std::cos(phi),
            sin_theta * std::sin(phi),
            std::cos(theta)};
}

std::pair<double, double> cart2sph(Vec3 const& v) {
    // returns (theta, phi) in radians
    double r = v.norm();
    if (r < 1e-10) return {0, 0}; // avoid division by zero
    double theta = std::acos(v.z / r); // polar angle
    double phi = std::atan2(v.y, v.x); // azimuthal angle
    return {theta, phi};
}

Mat3 rotation_matrix(Vec3 const& ax, double angle) {
    // rotates vectors around axis ax by angle in radians
    double c = std::cos(angle);
    double s = std::sin(angle);
    double t = 1.0 - c;
    double x = ax.x, y = ax.y, z = ax.z;
    double x2 = x * x, y2 = y * y, z2 = z * z;
    double xy = x * y, xz = x * z, yz = y * z;
    Mat3 R = Mat3 {
        {{t * x2 + c, t * xy - s * z, t * xz + s * y},
         {t * xy + s * z, t * y2 + c, t * yz - s * x},
         {t * xz - s * y, t * yz + s * x, t * z2 + c}}
    };
    return R;
}

Mat3 rot2gal(Vec3 const& LOS_dir, Vec3 const& src_dir) {
    // returns the rotation matrix that transforms vectors from the jet frame (th_emj, phi_emj)
    // to the galactic frame

    // LOS_dir is the direction of the LOS in the jet frame (th_view, phi_view)
    // src_dir is the direction of the source in the global frame (theta_src, phi_src)

    Vec3 z_prime = {0, 0, 1};

    // R(z', z) rotates vectors in the same manner as the rotation from z' to z
    // R^{-1}(z', z) = R(z, z') transforms vector components from z' frame to z frame, leaving the vector unchanged
    Mat3 R1 = RotA2B(LOS_dir, z_prime);
    Mat3 R2 = RotA2B(z_prime, -src_dir);
    return R2 * R1;
}