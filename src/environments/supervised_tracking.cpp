//
// Created by Khurram Javed on 2022-02-02.
//

#include "../../include/environments/supervised_tracking.h"
#include <iostream>
SupervisedTracking::SupervisedTracking(float input_mean,
                                       float input_std,
                                       float target_weight_mean,
                                       float target_weight_std,
                                       int dimensions,
                                       int seed) : weight_change_index_sampler(0, dimensions - 1),
                                                   input_sampler(0, 1),
                                                   input_mean(input_mean),
                                                   input_std(input_std),
                                                   target_mean(target_weight_mean),
                                                   target_std(target_weight_std), mt(seed),
                                                   dimension(dimensions),
                                                   time(0) {

  for (int c = 0; c < dimensions; c++) {
    this->target_weights.push_back(1);
  }
  base_x = this->generate_random_x();
}

void SupervisedTracking::change_target() {
  int index = weight_change_index_sampler(this->mt);
  this->target_weights[index] *= -1;
}

std::vector<float> SupervisedTracking::generate_random_x() {
  std::vector<float> x;
  x.reserve(dimension);
  for (int c = 0; c < dimension; c++) {
    x.push_back(input_sampler(this->mt));
  }
  return x;
}

std::vector<float> SupervisedTracking::step() {
  time++;
  if (time % 20 == 0)
    change_target();
  base_x = this->generate_random_x();
  std::vector<float> x;
  x.reserve(base_x.size());
  for (int counter = 0; counter < base_x.size(); counter++)
    x.push_back(base_x[counter] * input_std + input_mean);
  return x;
}

float SupervisedTracking::get_y() {
  float y = 0;
  for (int c = 0; c < dimension; c++) {
    y += this->target_weights[c] * base_x[c] * target_std;
  }
//  std::cout << y << std::endl;
  y += target_mean;
//  std::cout << "Target mena " << y <<std::endl;
  return y;
}

SupervisedLearning::SupervisedLearning(int dimensions,
                                       int seed, float target_noise) : mt(seed), weight_change_index_sampler(0, 1),
                                                   dimension(dimensions),
                                                   time(0), target_noise_sampler(-target_noise, target_noise) {


  for (int c = 0; c < dimensions; c++) {
    int sample = weight_change_index_sampler(this->mt);
    if (sample == 0)
      this->target_weights.push_back(1);
    else if (sample == 1)
      this->target_weights.push_back(-1);
  }
}

SupervisedLearningNormal::SupervisedLearningNormal(float input_mean, int dimensions, int seed, float target_noise) : SupervisedLearning(
    dimensions,
    seed, target_noise), input_sampler(input_mean, 1) {
  this->input_mean = input_mean;
}

SupervisedLearningNormalCapped::SupervisedLearningNormalCapped(float input_mean, int dimensions, int seed, float cap, float target_noise)
    : SupervisedLearningNormal(input_mean, dimensions, seed, target_noise) {
  this->cap = cap;
}

std::vector<float> SupervisedLearningNormalCapped::generate_random_x() {
  std::vector<float> x = SupervisedLearningNormal::generate_random_x();
  for (int c = 0; c < dimension; c++) {
    if (x[c] > this->input_mean + cap)
      x[c] = this->input_mean + cap;
    else if (x[c] < (this->input_mean - cap))
      x[c] = (this->input_mean - cap);
  }
  return x;
}

std::vector<float> SupervisedLearningNormal::generate_random_x() {
  std::vector<float> x;
  x.reserve(dimension);
  for (int c = 0; c < dimension; c++) {
    x.push_back(input_sampler(this->mt));
  }
  return x;
}

std::vector<float> SupervisedLearningNormal::step() {
  time++;
  base_x = this->generate_random_x();
  return base_x;
}

std::vector<float> SupervisedLearning::get_target_weights() {
  return target_weights;
}

float SupervisedLearning::get_y() {
  float y = 0;
  for (int c = 0; c < dimension; c++) {
    y += this->target_weights[c] * base_x[c];
  }
  y += target_noise_sampler(this->mt);
  return y;
}




SupervisedLearningBinary::SupervisedLearningBinary(float p, float input_mean, int dimensions, int seed, float target_noise)
    : SupervisedLearning(dimensions, seed, target_noise), inp_distribution(p) {
  p_val = p;
  this->input_mean = input_mean;
  k = 1 / sqrt(p_val * (1 - p_val));
}

std::vector<float> SupervisedLearningBinary::generate_random_x() {
  std::vector<float> x;
  x.reserve(dimension);
  for (int c = 0; c < dimension; c++) {
    x.push_back(inp_distribution(this->mt) * k - (k/2) + input_mean);
  }
  return x;
}

std::vector<float> SupervisedLearningBinary::step() {
  time++;
  base_x = this->generate_random_x();
  return base_x;
}




SupervisedLearningUniform::SupervisedLearningUniform(float a, float input_mean, int dimensions, int seed, float target_noise)
    : SupervisedLearning(dimensions, seed, target_noise), inp_distribution(a, a + sqrt(12)) {
  this->input_mean = input_mean;
  k = 1 / sqrt(p_val * (1 - p_val));
}

std::vector<float> SupervisedLearningUniform::generate_random_x() {
  std::vector<float> x;
  x.reserve(dimension);
  for (int c = 0; c < dimension; c++) {
    x.push_back(inp_distribution(this->mt)  + input_mean);
  }
  return x;
}

std::vector<float> SupervisedLearningUniform::step() {
  time++;
  base_x = this->generate_random_x();
  return base_x;
}



SupervisedLearningVariableBinary::SupervisedLearningVariableBinary(float p_start,
                                                                   float p_end,
                                                                   float input_mean,
                                                                   int dimensions,
                                                                   int seed, float target_noise) : SupervisedLearning(dimensions, seed, target_noise) {
  float increment = (p_end - p_start)/(dimensions-1);
  for(int c = 0; c < dimensions; c++){
    inp_distribution_vec.emplace_back(p_start + c*increment);
    float p_val  = p_start + c*increment;
    k_vector.push_back(1 / sqrt(p_val * (1 - p_val)));
  }
  this->input_mean = input_mean;
}


std::vector<float> SupervisedLearningVariableBinary::generate_random_x() {
  std::vector<float> x;
  x.reserve(dimension);
  for (int c = 0; c < dimension; c++) {
    x.push_back(inp_distribution_vec[c](this->mt) * k_vector[c] - (k_vector[c]/2) + input_mean);
  }
  return x;
}

std::vector<float> SupervisedLearningVariableBinary::step() {
  time++;
  base_x = this->generate_random_x();
  return base_x;
}










//

//
//SupervisedLearningBinary::SupervisedLearningBinary(float input_mean,
//                                       float input_std,
//                                       int dimensions,
//                                       int seed) : mt(seed), weight_change_index_sampler(0, 1),
//                                                   input_sampler(0, 1),
//                                                   input_mean(input_mean),
//                                                   input_std(input_std),
//                                                   dimension(dimensions),
//                                                   time(0){
//
//  for(int c = 0; c<dimensions; c++){
//    int sample = weight_change_index_sampler(this->mt);
//    if(sample == 0)
//      this->target_weights.push_back(1);
//    else if (sample == 1)
//      this->target_weights.push_back(-1);
//  }
//  base_x = this->generate_random_x();
//}
//
//
//
//std::vector<float> SupervisedLearningBinary::generate_random_x() {
//  std::vector<float> x;
//  x.reserve(dimension);
//  for(int c = 0; c<dimension; c++){
//    int val = weight_change_index_sampler(this->mt);
//    if(val == 0)
//      val = -1;
//    x.push_back(val + input_mean);
//  }
//  return x;
//}
//
//
//
//
//std::vector<float> SupervisedLearningBinary::get_target_weights() {
//  return target_weights;
//}
//
//float SupervisedLearningBinary::get_y() {
//  float y = 0;
//  for(int c = 0; c < dimension; c++){
//    y += this->target_weights[c]*(base_x[c]);
//  }
//  return y;
//}


// New one

