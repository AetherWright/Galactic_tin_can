#ifndef CPP_UTILS_HPP
#define CPP_UTILS_HPP

extern "C" {
    double cpp_logistic_growth(double n, double r, double k, int approximate);
    double cpp_distance(const double* p1, const double* p2, int dim);
}

#endif // CPP_UTILS_HPP
