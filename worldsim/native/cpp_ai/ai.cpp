#include "ai.hpp"
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <fstream>

struct SimplePerceptronHandle {
    std::vector<double> weights;
    double bias;
};

static double activation(double x) { return 1.0 / (1.0 + std::exp(-x)); }

SimplePerceptronHandle* sp_create(int n_inputs) {
    auto* h = new SimplePerceptronHandle();
    h->weights.resize(n_inputs);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i = 0; i < n_inputs; ++i) h->weights[i] = dist(gen);
    h->bias = dist(gen);
    return h;
}

void sp_destroy(SimplePerceptronHandle* h) { delete h; }

double sp_predict_prob(SimplePerceptronHandle* h, const double* inputs) {
    double s = h->bias;
    for (size_t i = 0; i < h->weights.size(); ++i) s += h->weights[i] * inputs[i];
    return activation(s);
}

int sp_predict(SimplePerceptronHandle* h, const double* inputs) {
    return sp_predict_prob(h, inputs) >= 0.5 ? 1 : 0;
}

void sp_train(SimplePerceptronHandle* h, const double* inputs, int label, double lr) {
    double out = sp_predict_prob(h, inputs);
    double err = label - out;
    for (size_t i = 0; i < h->weights.size(); ++i) {
        h->weights[i] += lr * err * inputs[i];
    }
    h->bias += lr * err;
}

struct QKeyHash {
    size_t operator()(const std::string& k) const noexcept {
        return std::hash<std::string>()(k);
    }
};

struct QLearningHandle {
    int n_inputs;
    int n_actions;
    double lr;
    double gamma;
    double epsilon = 0.1;
    double eps_decay = 0.995;
    int bins = 10;
    std::unordered_map<std::string, std::vector<double>, QKeyHash> table;
};

static std::string to_key(const double* state, int n_inputs, int bins) {
    std::ostringstream oss;
    for (int i = 0; i < n_inputs; ++i) {
        int v = static_cast<int>(state[i] * bins);
        oss << v << ',';
    }
    return oss.str();
}

QLearningHandle* ql_create(int n_inputs, int n_actions, double lr, double gamma) {
    auto* h = new QLearningHandle();
    h->n_inputs = n_inputs;
    h->n_actions = n_actions;
    h->lr = lr;
    h->gamma = gamma;
    h->epsilon = 0.1;
    h->eps_decay = 0.995;
    return h;
}

void ql_destroy(QLearningHandle* h) { delete h; }

static std::vector<double>& ensure_state(QLearningHandle* h, const std::string& key) {
    auto it = h->table.find(key);
    if (it == h->table.end()) {
        it = h->table.emplace(key, std::vector<double>(h->n_actions, 0.0)).first;
    }
    return it->second;
}

int ql_choose_action(QLearningHandle* h, const double* state) {
    std::string k = to_key(state, h->n_inputs, h->bins);
    auto& vec = ensure_state(h, k);
    int best = 0;
    for (int i = 1; i < h->n_actions; ++i) {
        if (vec[i] > vec[best]) best = i;
    }
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    int action;
    if (dist(gen) < h->epsilon) {
        action = std::uniform_int_distribution<>(0, h->n_actions - 1)(gen);
    } else {
        action = best;
    }
    h->epsilon *= h->eps_decay;
    if (h->epsilon < 0.01) h->epsilon = 0.01;
    return action;
}

void ql_train(QLearningHandle* h, const double* state, int action, double reward, const double* next_state) {
    std::string k = to_key(state, h->n_inputs, h->bins);
    std::string nk = to_key(next_state, h->n_inputs, h->bins);
    auto& cur = ensure_state(h, k);
    auto& nxt = ensure_state(h, nk);
    double max_next = nxt[0];
    for (int i = 1; i < h->n_actions; ++i) if (nxt[i] > max_next) max_next = nxt[i];
    double old = cur[action];
    cur[action] = old + h->lr * (reward + h->gamma * max_next - old);
}

void ql_predict(QLearningHandle* h, const double* state, double* out_q) {
    std::string k = to_key(state, h->n_inputs, h->bins);
    auto& vec = ensure_state(h, k);
    for (int i = 0; i < h->n_actions; ++i) out_q[i] = vec[i];
}

void ql_save(QLearningHandle* h, const char* path) {
    std::ofstream out(path);
    if (!out.is_open()) return;
    for (const auto& kv : h->table) {
        out << kv.first << ":\n";
        for (double v : kv.second) {
            out << "  - " << v << "\n";
        }
    }
}

void ql_load(QLearningHandle* h, const char* path) {
    std::ifstream in(path);
    if (!in.is_open()) return;
    h->table.clear();
    std::string line;
    std::string key;
    std::vector<double> vals;
    auto flush = [&]() {
        if (!key.empty()) {
            h->table[key] = vals;
            vals.clear();
        }
    };
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line.back() == ':') {
            flush();
            key = line.substr(0, line.size() - 1);
        } else {
            size_t pos = line.find('-');
            if (pos != std::string::npos) {
                double v = std::stod(line.substr(pos + 1));
                vals.push_back(v);
            }
        }
    }
    flush();
}

