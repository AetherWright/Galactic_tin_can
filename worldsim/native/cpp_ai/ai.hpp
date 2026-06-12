#ifndef CPP_AI_HPP
#define CPP_AI_HPP

#include <vector>
#include <unordered_map>
#include <string>

extern "C" {

struct SimplePerceptronHandle;
struct QLearningHandle;

SimplePerceptronHandle* sp_create(int n_inputs);
void sp_destroy(SimplePerceptronHandle* h);
double sp_predict_prob(SimplePerceptronHandle* h, const double* inputs);
int sp_predict(SimplePerceptronHandle* h, const double* inputs);
void sp_train(SimplePerceptronHandle* h, const double* inputs, int label, double lr);

QLearningHandle* ql_create(int n_inputs, int n_actions, double lr, double gamma);
void ql_destroy(QLearningHandle* h);
int ql_choose_action(QLearningHandle* h, const double* state);
void ql_train(QLearningHandle* h, const double* state, int action, double reward, const double* next_state);
void ql_predict(QLearningHandle* h, const double* state, double* out_q);
void ql_save(QLearningHandle* h, const char* path);
void ql_load(QLearningHandle* h, const char* path);
}

#endif // CPP_AI_HPP
