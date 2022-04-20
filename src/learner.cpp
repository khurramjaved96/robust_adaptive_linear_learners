//
// Created by Khurram Javed on 2022-02-02.
//

#include "../include/learner.h"
#include <math.h>
#include <iostream>
#include <tgmath.h>

LMS::LMS(float step_size, int d) : Learner(step_size, d) {
  target_test_mean = 0;
  bias_step_size = step_size;
  this->counter = 0;
}

Learner::Learner(float step_size, int d) {
  for (int counter = 0; counter < d; counter++) {
    weights.push_back(0);
    step_sizes.push_back(step_size);
    gradients.push_back(0);
  }
  step_size_normalization = 1;
  bias_weight = 0;
  bias_gradient = 0;
  dim = d;
}

float LMS::forward(std::vector<float> x) {
  this->counter++;
  float pred = 0;
  step_size_normalization =
      1.0 / (this->counter) * ((this->get_dot_product(x) + 1) + (counter - 1) * step_size_normalization);
  for (int counter = 0; counter < dim; counter++) {
    pred += weights[counter] * x[counter];
  }
  pred += bias_weight;
  return pred;
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
  float prod = 1;
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
    weights[c] -= (step_sizes[c] / step_size_normalization) * gradients[c];
  }
  bias_weight -= (bias_step_size / step_size_normalization) * bias_gradient;
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
  step_size_normalization =
      1.0 / (this->counter) * ((this->get_dot_product(x) + 1) + (counter - 1) * step_size_normalization);
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
    weights[c] -= (step_sizes[c] / step_size_normalization) * gradients[c];
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


//
//LMS_Input_Normalization::LMS_Input_Normalization(float step_size, int d) : LMS(step_size, d) {
//  for (int counter = 0; counter < d; counter++) {
//    input_normalization_mean.push_back(0);
//    input_normalization_std.push_back(1);
//    target_normalization_mean.push_back(0);
//    target_normalization_std.push_back(1);
//  }
//}
//
//float LMS_Input_Normalization::forward(std::vector<float> x) {
//  update_normalization_estimates(x);
//  x = normalize_x(x);
//  return LMS::forward(x);
//}
//
//std::vector<float> LMS_Input_Normalization::normalize_x(std::vector<float> x) {
//  for (int c = 0; c < dim; c++) {
//    x[c] = (x[c] - input_normalization_mean[c]) / input_normalization_std[c];
//  }
//  return x;
//}
//
//void LMS_Input_Normalization::update_normalization_estimates(std::vector<float> x) {
//  for (int c = 0; c < dim; c++) {
//    float old_mean = input_normalization_mean[c];
//    input_normalization_mean[c] = input_normalization_mean[c] * 0.9999 + x[c] * 0.0001;
//    input_normalization_std[c] = input_normalization_std[c] * 0.9999
//        + sqrt((x[c] - input_normalization_mean[c]) * (x[c] - old_mean)) * 0.0001;
//  }
//}
//
//void LMS_Input_Normalization::backward(std::vector<float> x, float pred, float target) {
//  float error = target - pred;
//  x = normalize_x(x);
//  LMS::backward(x, error);
//}
//
//std::vector<float> LMS_Input_Normalization::get_input_mean() {
//  return this->input_normalization_mean;
//}
//
//std::vector<float> LMS_Input_Normalization::get_input_std() {
//  return this->input_normalization_std;
//}
//
//LMS_Input_target_normalization::LMS_Input_target_normalization(float step_size, int d) : LMS_Input_Normalization(
//    step_size,
//    d) {
//  target_mean = 0;
//  target_std = 1;
//}
//
//float LMS_Input_target_normalization::forward(std::vector<float> x) {
//  float pred = LMS_Input_Normalization::forward(x);
////  std::cout << "Target std " << target_std << std::endl;
////  std::cout << "target mean " << target_mean << std::endl;
//  pred = (pred * target_std) + target_mean;
//  return pred;
//}
//
//void LMS_Input_target_normalization::backward(std::vector<float> x, float pred, float target) {
//  float target_orig = target;
//  target = (target - target_mean) / target_std;
////  std::cout << "Target normalized = " << target << std::endl;
////  Update target mean and std
//  pred = (pred - target_mean) / target_std;
//  target_test_mean = target_test_mean*0.9999 + 0.0001*target;
//  target_test_std = target_test_std*0.9999 + 0.0001*(target - target_test_mean)*(target - target_test_mean);
////  std::cout << "Target " << target << std::endl;
////  std::cout << "Target mean " << target_test_mean << " " << target_mean << std::endl;
////  std::cout << "Target std " << target_test_std << " " << target_std << std::endl;
//  LMS_Input_Normalization::backward(x, pred, target);
//  update_target_statistics(target_orig);
//}
//
//void LMS_Input_target_normalization::update_target_statistics(float target) {
////  std::cout << "Target " << target << std::endl;
//  float old_mean = target_mean;
//  target_mean = target_mean * 0.9999 + 0.0001 * target;
//  target_std = target_std * 0.9999 + sqrt((target - target_mean) * (target - old_mean)) * 0.0001;
//}
//
//IDBD::IDBD(float meta_step_size, int d) : LMS_Input_target_normalization(1e-5, d) {
//  for (int c = 0; c < d; c++) {
//    this->B.push_back(log(1e-4));
//    this->step_size_graidents.push_back(0);
//    this->h.push_back(0);
//    this->step_sizes[c] = exp(this->B[c]);
//  }
//  this->meta_step_size = meta_step_size;
//}
//
//float IDBD::forward(std::vector<float> x) {
//  return LMS_Input_target_normalization::forward(x);
//}
//
//void IDBD::backward(std::vector<float> x, float pred, float target) {
//
//  float target_norm = (target - target_mean)/target_std;
////  Update target mean and std
//  float pred_norm = (pred - target_mean)/target_std;
//  float error_norm = target_norm - pred_norm;
//  auto x_temp = LMS_Input_Normalization::normalize_x(x);
//  for(int c = 0; c<dim; c++){
//    this->step_size_graidents[c] = error_norm*x_temp[c]*h[c];
//    this->step_sizes[c] = exp(this->B[c]);
//  }
////  std::cout << error_norm << " " << x_temp[0] << std::endl;
////  std::cout << meta_step_size*error_norm*x_temp[0]*h[0] << std::endl;
////  std::cout << step_sizes[1] << std::endl;
//  LMS_Input_target_normalization::backward(x, pred, target);
//  for(int c = 0; c<dim; c++){
//    h[c] = h[c]*(1-step_sizes[c]*x_temp[c]*x_temp[c]) + step_sizes[c]*error_norm*x_temp[c];
//  }
//}
//
//void IDBD::update_parameters() {
//  for(int c = 0; c<dim; c++){
//    this->B[c] += meta_step_size*step_size_graidents[c];
//    this->step_sizes[c] = exp(this->B[c]);
////    std::cout << "C " << c << " " << this->step_sizes[c] << std::endl;
//  }
//
//  LMS::update_parameters();
//}
//

//
//
//IDBD::IDBDNormalized(float meta_step_size, int d) : LMS_Input_target_normalization(1e-5, d) {
//  for (int c = 0; c < d; c++) {
//    this->B.push_back(log(1e-4));
//    this->step_size_graidents.push_back(0);
//    this->h.push_back(0);
//    this->step_sizes[c] = exp(this->B[c]);
//  }
//  this->meta_step_size = meta_step_size;
//}
//
//float IDBDNormalized::forward(std::vector<float> x) {
//  return LMS_Input_Normalization::forward(x);
//}

//void IDBDNormalized::backward(std::vector<float> x, float pred, float target) {
//
//  float target_norm = (target - target_mean)/target_std;
////  Update target mean and std
//  float pred_norm = (pred - target_mean)/target_std;
//  float error_norm = target_norm - pred_norm;
//  auto x_temp = LMS_Input_Normalization::normalize_x(x);
//  for(int c = 0; c<dim; c++){
//    this->step_size_graidents[c] = error_norm*x_temp[c]*h[c];
//    this->step_sizes[c] = exp(this->B[c]);
//  }
////  std::cout << error_norm << " " << x_temp[0] << std::endl;
////  std::cout << meta_step_size*error_norm*x_temp[0]*h[0] << std::endl;
////  std::cout << step_sizes[1] << std::endl;
//  LMS_Input_target_normalization::backward(x, pred, target);
//  for(int c = 0; c<dim; c++){
//    h[c] = h[c]*(1-step_sizes[c]*x_temp[c]*x_temp[c]) + step_sizes[c]*error_norm*x_temp[c];
//  }
//}
//
//void IDBDNormalized::update_parameters() {
//  for(int c = 0; c<dim; c++){
//    this->B[c] += meta_step_size*step_size_graidents[c];
//    this->step_sizes[c] = exp(this->B[c]);
////    std::cout << "C " << c << " " << this->step_sizes[c] << std::endl;
//  }
//
//  LMS::update_parameters();
//}