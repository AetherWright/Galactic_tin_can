#include "utils.hpp"
#include <cmath>

extern "C" {

double cpp_logistic_growth(double n, double r, double k, int approximate) {
    if (approximate) return n + r * n;
    return n + r * n * (1.0 - n / k);
}

double cpp_distance(const double* p1, const double* p2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        double d = p1[i] - p2[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

}
