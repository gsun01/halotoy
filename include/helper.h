#pragma once

#include <functional>
#include <string>
#include <tuple>
#include <sstream>
#include <cmath>

// signum function
template <typename T>
inline int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// to_string with fixed precision
inline std::string to_string(double value, int precision = 2) {
    std::ostringstream out;
    out.precision(precision);
    out << std::fixed << value;
    return out.str();
}

// 3D vector
struct Vec3 {
    double x, y, z;
    inline Vec3 operator+(Vec3 const& o) const { return {x+o.x, y+o.y, z+o.z}; }
    inline Vec3 operator-(Vec3 const& o) const { return {x-o.x, y-o.y, z-o.z}; }
    inline Vec3 operator-() const { return {-x, -y, -z}; }
    inline Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    inline Vec3 cross(Vec3 const& o) const {
        return { y*o.z - z*o.y,
                 z*o.x - x*o.z,
                 x*o.y - y*o.x };
    }
    inline double dot(Vec3 const& o) const {
        return x*o.x + y*o.y + z*o.z;
    }
    inline double norm() const {
        return std::sqrt(x*x + y*y + z*z);
    }
    inline Vec3 normalized() const {
        double n = norm();
        return (n > 1e-10) ? Vec3{x/n, y/n, z/n} : Vec3{0,0,0};
    }
};
inline Vec3 operator*(double s, Vec3 const& v) {
    return {v.x*s, v.y*s, v.z*s};
}

// 3x3 matrix
struct Mat3 {
    double m[3][3];
    inline Vec3 operator*(Vec3 const& v) const {
        return {m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
                m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
                m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z};
    }
    inline Mat3 operator+(Mat3 const& o) const {
        Mat3 res{};
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            res.m[i][j] = m[i][j] + o.m[i][j];
        return res;
    }
    inline Mat3 operator-(Mat3 const& o) const {
        Mat3 res{};
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            res.m[i][j] = m[i][j] - o.m[i][j];
        return res;
    }
    inline Mat3 operator*(Mat3 const& o) const {
        Mat3 res{};
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) for(int k=0;k<3;k++)
            res.m[i][j] += m[i][k] * o.m[k][j];
        return res;
    }
    inline Mat3 operator*(double s) const {
        Mat3 res{};
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            res.m[i][j] = m[i][j] * s;
        return res;
    }
    inline Mat3 operator-() const {
        Mat3 res{};
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            res.m[i][j] = -m[i][j];
        return res;
    }
};
inline Mat3 operator*(double s, Mat3 const& o) {
    Mat3 res{};
    for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        res.m[i][j] = s * o.m[i][j];
    return res;
}

// identity matrix
inline Mat3 id_mat() {
    return Mat3{{{1,0,0},{0,1,0},{0,0,1}}};
}

// spherical -> Cartesian
inline Vec3 sph2cart(double theta, double phi) {
    double st = std::sin(theta);
    return {st*std::cos(phi), st*std::sin(phi), std::cos(theta)};
}
inline std::tuple<double,double> cart2sph(Vec3 const& v) {
    double r = v.norm();
    if (r < 1e-10) return {0,0};
    return {std::acos(v.z/r), std::atan2(v.y, v.x)};
}

//----------------------------------------
// Declarations for larger functions
//----------------------------------------

double adaptiveGaussQuadrature(std::function<double(double)> func,
                               double a, double b,
                               double tol = 1e-6);

double bisection(std::function<double(double)> F_func,
                 double low, double high,
                 double tol = 1e-6, int max_iter = 1000);

double F(double z_s, double z_E, double E);

double calc_z(double E, double z_E, double low, double high);

Mat3 rotation_matrix(Vec3 const& ax, double angle);

Mat3 RotA2B(Vec3 const& a, Vec3 const& b);

Mat3 jet2gal(Vec3 const& LOS_dir, Vec3 const& src_dir);