//
// Created by Khurram Javed on 2022-02-02.
//

#include "../include/learner.h"
#include <math.h>
#include <iostream>
#include <tgmath.h>

LMS::LMS(float step_size, int d) : Learner(step_size, d) {
  bias_step_size = step_size;
  this->counter = 0;
}

Learner::Learner(float step_size, int d) {
  for (int counter = 0; counter < d; counter++) {
    weights.push_back(0);
    step_sizes.push_back(step_size);
    gradients.push_back(0);
  }
  bias_weight = 0;
  bias_gradient = 0;
  dim = d;
}

float LMS::forward(std::vector<float> x) {
  this->counter++;
  float pred = 0;
  for (int counter = 0; counter < dim; counter++) {
    pred += weights[counter] * x[counter];
  }
  pred += bias_weight;
  return pred;
}

float LMSNormalizedStepSize::forward(std::vector<float> x) {
  float pred = LMS::forward(x);
  step_size_normalization = step_size_normalization * 0.9995 + 0.0005 * (this->get_dot_product(x) + 1);
  return pred;
}

LMSNormalizedStepSize::LMSNormalizedStepSize(float step_size, int d) : LMS(step_size, d) {
  step_size_normalization = 1;
};

LMSNormalizedInputsAndStepSizes::LMSNormalizedInputsAndStepSizes(float step_size, int d) : LMSNormalizedStepSize(
    step_size,
    d) {
  for (int counter = 0; counter < d; counter++) {
    input_normalization_mean.push_back(0);
    input_normalization_std.push_back(1);
  }
}

float LMSNormalizedInputsAndStepSizes::forward(std::vector<float> x) {
  this->counter++;
  update_normalization_estimates(x);
  x = normalize_x(x);
  float pred = LMSNormalizedStepSize::forward(x);
  return pred;
}

void LMSNormalizedInputsAndStepSizes::update_normalization_estimates(std::vector<float> x) {
  for (int c = 0; c < dim; c++) {
    input_normalization_mean[c] = input_normalization_mean[c] * 0.9995 + 0.0005 * x[c];
    input_normalization_std[c] = input_normalization_std[c] * 0.9995
        + 0.0005 * (input_normalization_mean[c] - x[c]) * (input_normalization_mean[c] - x[c]);
  }
}

std::vector<float> LMSNormalizedInputsAndStepSizes::normalize_x(std::vector<float> x) {
  for (int c = 0; c < dim; c++) {
//    std::cout << "Before " << x[c] << std::endl;
    x[c] = (x[c] - input_normalization_mean[c]) / sqrt(input_normalization_std[c]);
//    std::cout << "Before " << x[c] << std::endl;
  }

  return x;
}

void LMSNormalizedInputsAndStepSizes::backward(std::vector<float> x, float pred, float target) {
  float error = target - pred;
  x = normalize_x(x);
  for (int counter = 0; counter < dim; counter++) {
    gradients[counter] -= x[counter] * error;
  }

  bias_gradient -= error;
}

void LMSNormalizedStepSize::update_parameters() {
  for (int c = 0; c < dim; c++) {
    weights[c] -= (step_sizes[c] / step_size_normalization) * gradients[c];
  }
  bias_weight -= (bias_step_size / step_size_normalization) * bias_gradient;
}

float NormalizedLMS::forward(std::vector<float> x) {
  cur_dot_product = get_dot_product(x);
  return LMS::forward(x);
}

NormalizedLMS::NormalizedLMS(float step_size, int d) : LMS(step_size, d) {
  cur_dot_product = 0;
}

float Learner::get_dot_product(std::vector<float> my_vec) {
//  Prod = 1 to take into account the bias term
  float prod = 0;
  for (int c = 0; c < my_vec.size(); c++) {
    prod += my_vec[c] * my_vec[c];
  }
  return prod;
}

void LMS::backward(std::vector<float> x, float pred, float target) {
  float error = target - pred;
  for (int counter = 0; counter < dim; counter++) {
    gradients[counter] -= x[counter] * error;
  }
  bias_gradient -= error;
}

void Learner::zero_grad() {
  for (int c = 0; c < dim; c++) {
    gradients[c] = 0;
  }
  bias_gradient = 0;
}

float Learner::distance_to_target_weights(std::vector<float> target_weights) {
  float avg_distance = 0;
  for (int c = 0; c < dim; c++) {
    avg_distance += (target_weights[c] - weights[c]) * (target_weights[c] - weights[c]);
  }
  return avg_distance / dim;
}

void LMS::update_parameters() {
  for (int c = 0; c < dim; c++) {
    weights[c] -= (step_sizes[c]) * gradients[c];
  }
  bias_weight -= (bias_step_size) * bias_gradient;
}

void NormalizedLMS::update_parameters() {
  for (int c = 0; c < dim; c++) {
    weights[c] -= (step_sizes[c] / cur_dot_product) * gradients[c];
  }
  bias_weight -= (bias_step_size / cur_dot_product) * bias_gradient;
}

Nadaline::Nadaline(float step_size, int d) : Learner(step_size, d) {
  for (int counter = 0; counter < d; counter++) {
    input_normalization_mean.push_back(0);
    input_normalization_std.push_back(1);
  }
  this->counter = 0;
  dim = d;
}

float Nadaline::forward(std::vector<float> x) {
  this->counter++;
  update_normalization_estimates(x);
  x = normalize_x(x);
//  step_size_normalization =
//      1.0 / (this->counter) * ((this->get_dot_product(x) + 1) + (counter - 1) * step_size_normalization);
  float pred = 0;
  for (int counter = 0; counter < dim; counter++) {
    pred += weights[counter] * x[counter];
  }
  pred += bias_weight;
  return pred;
}

void Nadaline::backward(std::vector<float> x, float pred, float target) {
  x = normalize_x(x);
  float error = target - pred;
  for (int counter = 0; counter < dim; counter++) {
    gradients[counter] -= x[counter] * error;
  }
  bias_weight = (1.0f) / counter * (target + (counter - 1) * bias_weight);
//  std::cout << "Bias weight = " << bias_weight << std::endl;
}

void Nadaline::update_parameters() {
  for (int c = 0; c < dim; c++) {
    weights[c] -= (step_sizes[c]) * gradients[c];
  }
}

void Nadaline::update_normalization_estimates(std::vector<float> x) {
  for (int c = 0; c < dim; c++) {
    input_normalization_mean[c] = 1.0 / (this->counter) * (x[c] + (counter - 1) * input_normalization_mean[c]);
    input_normalization_std[c] = 1.0;
  }
}

std::vector<float> Nadaline::normalize_x(std::vector<float> x) {
  for (int c = 0; c < dim; c++) {
    x[c] = (x[c] - input_normalization_mean[c]) / sqrt(input_normalization_std[c]);
  }
  return x;
}

std::vector<float> Learner::get_weights() {
  return weights;
}

RMSPropLMS::RMSPropLMS(float step_size, int d, float beta, float epsilon) : LMS(step_size, d) {
    this->beta = beta;
    this->epsilon = epsilon;
    this->v_bias = 0;
    for (int i = 0; i < dim; i++) {
        this->v.push_back(0);
    }
}

void RMSPropLMS::update_RMSProp_statistics() {
    for (int i = 0; i < dim; i++) {
        this->v[i] = this->v[i] * this->beta + (1 - this->beta) * this->gradients[i] * this->gradients[i];
    }
    this->v_bias = this->v_bias * this->beta + (1 - this->beta) * this->bias_gradient;
}

void RMSPropLMS::update_parameters() {
    this->update_RMSProp_statistics();
    for (int i = 0; i < dim; i++) 
        weights[i] -= this->step_sizes[i] * gradients[i] / (sqrt(this->v[i]) + this->epsilon);
    
    bias_weight -= this->bias_step_size * bias_gradient / (sqrt(this->v_bias) + this->epsilon);
}

AdamLMS::AdamLMS(float step_size, int d, float b1, float b2, float epsilon) : LMS(step_size, d) {
  this->b1 = b1;
  this->b2 = b2;
  m1_bias = 0;
  m2_bias = 0;
  this->epsilon = epsilon;
  for (int c = 0; c < dim; c++) {
    this->m1.push_back(0);
    this->m2.push_back(0);
  }
}

void AdamLMS::update_adam_statistics() {
  for (int c = 0; c < dim; c++) {
    this->m1[c] = this->m1[c] * this->b1 + (1 - this->b1) * this->gradients[c];
    this->m2[c] = this->m2[c] * this->b2 + (1 - this->b2) * this->gradients[c] * this->gradients[c];
  }
  m1_bias = m1_bias * this->b1 + (1 - this->b1) * bias_gradient;
  m2_bias = m2_bias * this->b2 + (1 - this->b2) * bias_gradient * bias_gradient;
}

void AdamLMS::update_parameters() {
  this->update_adam_statistics();
  for (int c = 0; c < dim; c++) {
    float m1_hat = m1[c] / (1 - pow(b1, this->counter));
    float m2_hat = m2[c] / (1 - pow(b2, this->counter));
    weights[c] -= (step_sizes[c]) * m1_hat / (sqrt(m2_hat) + epsilon);
  }
  float m1_bias_hat = m1_bias / (1 - pow(b1, this->counter));
  float m2_bias_hat = m2_bias / (1 - pow(b2, this->counter));
  bias_weight -= (bias_step_size * m1_bias_hat) / (sqrt(m2_bias_hat) + epsilon);
}

NormalizedIDBD::NormalizedIDBD(float meta_step_size, float step_size, int d)
    : LMSNormalizedInputsAndStepSizes(step_size, d) {
  for (int c = 0; c < d; c++) {
    this->B.push_back(log(step_size));
    this->step_size_gradients.push_back(0);
    this->h.push_back(0);
    this->step_sizes[c] = exp(this->B[c]);
  }
  this->meta_step_size = meta_step_size;
}

void NormalizedIDBD::backward(std::vector<float> x, float pred, float target) {

  float error = target - pred;
  auto unnorm_x = x;
  x = LMSNormalizedInputsAndStepSizes::normalize_x(x);
  for (int c = 0; c < dim; c++) {
    this->B[c] += meta_step_size * error * x[c] * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - (step_sizes[c] / step_size_normalization) * x[c] * x[c]);
    if (temp > 0)
      h[c] = h[c] * temp + (step_sizes[c] / step_size_normalization) * error * x[c];
    else
      h[c] = (step_sizes[c] / step_size_normalization) * error * x[c];
  }
  LMSNormalizedInputsAndStepSizes::backward(unnorm_x, pred, target);
}

IDBD::IDBD(float meta_step_size, float step_size, int d) : LMS(step_size, d) {
  for (int c = 0; c < d; c++) {
    this->B.push_back(log(step_size));
    this->step_size_gradients.push_back(0);
    this->h.push_back(0);
    this->step_sizes[c] = exp(this->B[c]);
  }
  h_bias = 0;
  B_bias = log(step_size);
  bias_step_size = exp(B_bias);
  this->meta_step_size = meta_step_size;
}

IDBDBetaNorm::IDBDBetaNorm(float meta_step_size, float step_size, int d) : LMS(step_size, d) {
  for (int c = 0; c < d; c++) {
    this->B.push_back(log(step_size));
    this->step_size_gradients.push_back(0);
    this->h.push_back(0);
    this->step_sizes[c] = exp(this->B[c]);
  }
  h_bias = 0;
  B_bias = log(step_size);
  bias_step_size = exp(B_bias);
  this->meta_step_size = meta_step_size;
  for (int c = 0; c < dim; c++) {
    std_delta.push_back(1);
    mean_delta.push_back(0);
  }
  std_bias_delta = 1;
  mean_bias_delta = 0;
}

IDBDBest::IDBDBest(float meta_step_size, float step_size, int d) : LMSNormalizedInputsAndStepSizes(step_size, d) {
  for (int c = 0; c < d; c++) {
    this->B.push_back(log(step_size));
    this->step_size_gradients.push_back(0);
    this->h.push_back(0);
    this->step_sizes[c] = exp(this->B[c]);
  }
  for (int c = 0; c < dim; c++) {
    std_delta.push_back(1);
    mean_delta.push_back(0);
  }
  std_bias_delta = 1;
  mean_bias_delta = 0;
  h_bias = 0;
  B_bias = log(step_size);
  bias_step_size = exp(B_bias);
  this->meta_step_size = meta_step_size;
}

void IDBDBest::backward(std::vector<float> x, float pred, float target) {
  auto x_normalized = normalize_x(x);
  float error = target - pred;
  for (int c = 0; c < dim; c++) {
    float g = error * x_normalized[c];
    mean_delta[c] = mean_delta[c] * 0.9995 + 0.0005 * g;
    std_delta[c] = std_delta[c] * 0.9995 + 0.0005 * (mean_delta[c] - g) * (mean_delta[c] - g);
    float norm_g = (g) / float(sqrt(std_delta[c]) + 0.0001);
    this->B[c] += (meta_step_size * step_size_normalization) * norm_g * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - (step_sizes[c] / step_size_normalization) * x_normalized[c] * x_normalized[c]);
    if (temp > 0)
      h[c] = h[c] * temp + (step_sizes[c] / step_size_normalization) * norm_g;
    else
      h[c] = (step_sizes[c] / step_size_normalization) * norm_g;
  }
  float g = error * 1;
  mean_bias_delta = mean_bias_delta * 0.9995 + 0.0005 * g;
  std_bias_delta = std_bias_delta * 0.9995 + 0.0005 * (mean_bias_delta - g) * (mean_bias_delta - g);
  float norm_g = (g) / (sqrt(std_bias_delta) + 0.0001);
  B_bias += (meta_step_size * step_size_normalization) * norm_g * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - (bias_step_size / step_size_normalization));
  if (temp > 0)
    h_bias = h_bias * temp + (bias_step_size / step_size_normalization) * norm_g;
  else
    h_bias = (bias_step_size / step_size_normalization) * norm_g;

  for (int i = 0; i < dim; i++) {
    if (step_sizes[i] > 1.0) {
      step_sizes[i] = 0.98;
      h[i] = 0;
    }
//    step_sizes[i] /= (sum_of_steps * (1.0 / thred));
  }
  if (bias_step_size > 1.0) {
    h_bias = 0;
    bias_step_size = 0.98;
  }
//
  LMSNormalizedInputsAndStepSizes::backward(x, pred, target);
}

IDBDBestYNorm::IDBDBestYNorm(float meta_step_size, float step_size, int d) : LMSNormalizedInputsAndStepSizes(step_size,
                                                                                                             d) {
  for (int c = 0; c < d; c++) {
    this->B.push_back(log(step_size));
    this->step_size_gradients.push_back(0);
    this->h.push_back(0);
    this->step_sizes[c] = exp(this->B[c]);
  }
  for (int c = 0; c < dim; c++) {
    std_delta.push_back(1);
    mean_delta.push_back(0);
  }
  std_bias_delta = 1;
  mean_bias_delta = 0;
  h_bias = 0;
  B_bias = log(step_size);
  bias_step_size = exp(B_bias);
  this->meta_step_size = meta_step_size;
}

void IDBDBestYNorm::backward(std::vector<float> x, float pred, float target) {
  auto x_normalized = normalize_x(x);
  float error = target - pred;
  for (int c = 0; c < dim; c++) {
    float g = error * x_normalized[c];
    mean_delta[c] = mean_delta[c] * 0.9995 + 0.0005 * target;
    std_delta[c] = std_delta[c] * 0.9995 + 0.0005 * (mean_delta[c] - target) * (mean_delta[c] - target);
    float norm_g = (g) / float(sqrt(std_delta[c]) + 0.0001);
    this->B[c] += (meta_step_size * step_size_normalization) * norm_g * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - (step_sizes[c] / step_size_normalization) * x_normalized[c] * x_normalized[c]);
    if (temp > 0)
      h[c] = h[c] * temp + (step_sizes[c] / step_size_normalization) * norm_g;
    else
      h[c] = (step_sizes[c] / step_size_normalization) * norm_g;
  }
  float g = error * 1;
  mean_bias_delta = mean_bias_delta * 0.9995 + 0.0005 * target;
  std_bias_delta = std_bias_delta * 0.9995 + 0.0005 * (mean_bias_delta - target) * (mean_bias_delta - target);
  float norm_g = (g) / (sqrt(std_bias_delta) + 0.0001);
  B_bias += (meta_step_size * step_size_normalization) * norm_g * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - (bias_step_size / step_size_normalization));
  if (temp > 0)
    h_bias = h_bias * temp + (bias_step_size / step_size_normalization) * norm_g;
  else
    h_bias = (bias_step_size / step_size_normalization) * norm_g;

  for (int i = 0; i < dim; i++) {
    if (step_sizes[i] > 1.0) {
      step_sizes[i] = 0.98;
      h[i] = 0;
    }
//    step_sizes[i] /= (sum_of_steps * (1.0 / thred));
  }
  if (bias_step_size > 1.0) {
    h_bias = 0;
    bias_step_size = 0.98;
  }
//
  LMSNormalizedInputsAndStepSizes::backward(x, pred, target);
}

IDBDNorm::IDBDNorm(float meta_step_size, float step_size, int d) : LMSNormalizedInputsAndStepSizes(step_size,
                                                                                                   d) {
  for (int c = 0; c < d; c++) {
    this->B.push_back(log(step_size));
    this->step_size_gradients.push_back(0);
    this->h.push_back(0);
    this->step_sizes[c] = exp(this->B[c]);
  }
  for (int c = 0; c < dim; c++) {
    std_delta.push_back(1);
    mean_delta.push_back(0);
  }
  std_bias_delta = 1;
  mean_bias_delta = 0;
  h_bias = 0;
  B_bias = log(step_size);
  bias_step_size = exp(B_bias);
  this->meta_step_size = meta_step_size;
}

void IDBDNorm::backward(std::vector<float> x, float pred, float target) {
  auto x_normalized = normalize_x(x);
  float error = target - pred;
  for (int c = 0; c < dim; c++) {
    float g = error * x_normalized[c];
    mean_delta[c] = mean_delta[c] * 0.9995 + 0.0005 * target;
    std_delta[c] = std_delta[c] * 0.9995 + 0.0005 * (mean_delta[c] - target) * (mean_delta[c] - target);
    float norm_g = g;
    this->B[c] += (meta_step_size) * norm_g * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - (step_sizes[c] / step_size_normalization) * x_normalized[c] * x_normalized[c]);
    if (temp > 0)
      h[c] = h[c] * temp + (step_sizes[c] / step_size_normalization) * norm_g;
    else
      h[c] = (step_sizes[c] / step_size_normalization) * norm_g;
  }
  float g = error * 1;
  mean_bias_delta = mean_bias_delta * 0.9995 + 0.0005 * target;
  std_bias_delta = std_bias_delta * 0.9995 + 0.0005 * (mean_bias_delta - target) * (mean_bias_delta - target);
  float norm_g = g;
  B_bias += (meta_step_size) * norm_g * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - (bias_step_size / step_size_normalization));
  if (temp > 0)
    h_bias = h_bias * temp + (bias_step_size / step_size_normalization) * norm_g;
  else
    h_bias = (bias_step_size / step_size_normalization) * norm_g;

  for (int i = 0; i < dim; i++) {
    if (step_sizes[i] > 1.0) {
      step_sizes[i] = 0.98;
      this->B[i] = log(step_sizes[i]);
      h[i] = 0;
    }
//    step_sizes[i] /= (sum_of_steps * (1.0 / thred));
  }
  if (bias_step_size > 1.0) {
    h_bias = 0;
    bias_step_size = 0.98;
    B_bias = log(bias_step_size);
  }
//
  LMSNormalizedInputsAndStepSizes::backward(x, pred, target);
}

void IDBDBest::print_information(std::vector<float> x, float pred, float target) {
  auto x_normalized = normalize_x(x);
  float error = target - pred;
  for (int c = 0; c < 1; c++) {
    float g = error * x_normalized[c];
    mean_delta[c] = mean_delta[c] * 0.9995 + 0.0005 * g;
    std_delta[c] = std_delta[c] * 0.9995 + 0.0005 * (mean_delta[c] - g) * (mean_delta[c] - g);
    float norm_g = (g) / float(sqrt(std_delta[c]) + 0.0001);
    std::cout << "STD delta = " << float(sqrt(std_delta[c]) + 0.0001) << std::endl;
    std::cout << "step_size_normalization = " << step_size_normalization << std::endl;
    std::cout << "Step-size " << step_sizes[c] << std::endl;
  }

}

void IDBD::backward(std::vector<float> x, float pred, float target) {

  float error = target - pred;
  for (int c = 0; c < dim; c++) {
    this->B[c] += meta_step_size * error * x[c] * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - step_sizes[c] * x[c] * x[c]);
    if (temp > 0)
      h[c] = h[c] * temp + step_sizes[c] * error * x[c];
    else
      h[c] = step_sizes[c] * error * x[c];
  }

  B_bias += meta_step_size * error * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - bias_step_size);
  if (temp > 0)
    h_bias = h_bias * temp + bias_step_size * error;
  else
    h_bias = bias_step_size * error;
  LMS::backward(x, pred, target);
}

void IDBDBetaNorm::backward(std::vector<float> x, float pred, float target) {

  float error = target - pred;
  for (int c = 0; c < dim; c++) {
    float g = error * x[c] * h[c];
    mean_delta[c] = mean_delta[c] * 0.9995 + 0.0005 * g;
    std_delta[c] = std_delta[c] * 0.9995 + 0.0005 * (mean_delta[c] - g) * (mean_delta[c] - g);
    this->B[c] += (meta_step_size / sqrt(std_delta[c]) + 0.0001) * error * x[c] * h[c];

    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - step_sizes[c] * x[c] * x[c]);
    if (temp > 0)
      h[c] = h[c] * temp + step_sizes[c] * error * x[c];
    else
      h[c] = step_sizes[c] * error * x[c];
  }

  B_bias += meta_step_size * error * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - bias_step_size);
  if (temp > 0)
    h_bias = h_bias * temp + bias_step_size * error;
  else
    h_bias = bias_step_size * error;
  LMS::backward(x, pred, target);
}

NIDBD1::NIDBD1(float meta_step_size, float step_size, int d) : IDBD(meta_step_size,
                                                                    step_size,
                                                                    d) {
  std_delta = 1;
  mean_delta = 0;
}

void NIDBD1::backward(std::vector<float> x, float pred, float target) {

  float error = target - pred;
  mean_delta = mean_delta * 0.9995 + 0.0005 * error;
  std_delta = std_delta * 0.9995 + 0.0005 * (mean_delta - error) * (mean_delta - error);
  float normalized_error = (error - mean_delta) / (sqrt(std_delta) + 0.0001);
  for (int c = 0; c < dim; c++) {
    this->B[c] += meta_step_size * normalized_error * x[c] * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - step_sizes[c] * x[c] * x[c]);
    if (temp > 0)
      h[c] = h[c] * temp + step_sizes[c] * normalized_error * x[c];
    else
      h[c] = step_sizes[c] * normalized_error * x[c];
  }

  B_bias += meta_step_size * normalized_error * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - bias_step_size);
  if (temp > 0)
    h_bias = h_bias * temp + bias_step_size * normalized_error;
  else
    h_bias = bias_step_size * normalized_error;
  LMS::backward(x, pred, target);
}

NIDBD2::NIDBD2(float meta_step_size, float step_size, int d) : IDBD(meta_step_size,
                                                                    step_size,
                                                                    d) {
  for (int c = 0; c < dim; c++) {
    std_delta.push_back(1);
    mean_delta.push_back(0);
  }
  std_bias_delta = 1;
  mean_bias_delta = 0;
}

void NIDBD2::backward(std::vector<float> x, float pred, float target) {

  float error = target - pred;
  for (int c = 0; c < dim; c++) {
    float g = error * x[c];
    mean_delta[c] = mean_delta[c] * 0.9995 + 0.0005 * g;
    std_delta[c] = std_delta[c] * 0.9995 + 0.0005 * (mean_delta[c] - g) * (mean_delta[c] - g);
    float norm_g = (g - mean_delta[c]) / (sqrt(std_delta[c]) + 0.0001);
    this->B[c] += meta_step_size * norm_g * h[c];
    this->step_sizes[c] = exp(this->B[c]);
    float temp = (1 - step_sizes[c] * x[c] * x[c]);
    if (temp > 0)
      h[c] = h[c] * temp + step_sizes[c] * norm_g;
    else
      h[c] = step_sizes[c] * norm_g;
  }
  float g = error * 1;
  mean_bias_delta = mean_bias_delta * 0.9995 + 0.0005 * g;
  std_bias_delta = std_bias_delta * 0.9995 + 0.0005 * (mean_bias_delta - g) * (mean_bias_delta - g);
  float norm_g = (g - mean_bias_delta) / (sqrt(std_bias_delta) + 0.0001);
  B_bias += meta_step_size * norm_g * h_bias;
  bias_step_size = exp(B_bias);
  float temp = (1 - bias_step_size);
  if (temp > 0)
    h_bias = h_bias * temp + bias_step_size * norm_g;
  else
    h_bias = bias_step_size * norm_g;
  LMS::backward(x, pred, target);
}

