//
// Created by Khurram Javed on 2022-06-20.
//

#include "../include/idbd_variants.h"
#include "../include/learner.h"
#include <math.h>
#include <iostream>
#include <tgmath.h>
#include "../include/utils.h"

IDBDSmoothSteps::IDBDSmoothSteps(float meta_step_size, float step_size, int d, float percentange_change) : IDBD(
    meta_step_size,
    step_size,
    d) {
  max_effective_step_size = 1;
  min_effective_step_size = 1e-4;
  for (int i = 0; i < d; i++) {
    trace_meta_grad_features.push_back(1);
  }
  max_allowed_change = percentange_change;
  trace_meta_grad_bias = 1;
}

void IDBDSmoothSteps::backward(std::vector<float> x, float pred, float target) {

  float error = target - pred;
  float effective_step_size_before = 0;
  for (int c = 0; c < dim; c++) {
    effective_step_size_before += step_sizes[c];
  }
//  if (int(this->counter) % 10000 == 0) {
//    std::cout << "Effective step_size = " << effective_step_size_before << std::endl;
//    print_vector(step_sizes);
//
//  }

  auto old_step_sizes = step_sizes;
  auto old_B = B;
  for (int c = 0; c < dim; c++) {
    float grad_meta = error * x[c] * h[c];
    this->B[c] += (meta_step_size * grad_meta);
    this->step_sizes[c] = exp(this->B[c]);
  }

  for(int inner_c = 0; inner_c < dim; inner_c++){
    if(step_sizes[inner_c] > old_step_sizes[inner_c]){
      if(step_sizes[inner_c] - old_step_sizes[inner_c] > (max_effective_step_size - step_sizes[inner_c])*max_allowed_change){
        float max_change = ((max_effective_step_size - effective_step_size_before) * max_allowed_change);
        float actual_change = step_sizes[inner_c] - old_step_sizes[inner_c];
        float temp_meta_lr_change = max_change / actual_change;
        this->step_sizes[inner_c] =
            old_step_sizes[inner_c] + (step_sizes[inner_c] - old_step_sizes[inner_c]) * temp_meta_lr_change;
        this->B[inner_c] = log(this->step_sizes[inner_c]);
      }
    }
    else if(step_sizes[inner_c] < old_step_sizes[inner_c])
    {
      if(old_step_sizes[inner_c] - step_sizes[inner_c]
          > (old_step_sizes[inner_c] - min_effective_step_size) * max_allowed_change){

        float max_change = ((old_step_sizes[inner_c] - min_effective_step_size) * max_allowed_change);
        float actual_change = old_step_sizes[inner_c] - step_sizes[inner_c];
        float temp_meta_lr_change = max_change / actual_change;
        this->step_sizes[inner_c] =
            old_step_sizes[inner_c] + (step_sizes[inner_c] - old_step_sizes[inner_c]) * temp_meta_lr_change;
        this->B[inner_c] = log(this->step_sizes[inner_c]);
      }
    }
  }

  float effective_step_size_after = 0;
  for (int c = 0; c < dim; c++) {
    effective_step_size_after += step_sizes[c];
  }

  if (effective_step_size_after - effective_step_size_before
      > (max_effective_step_size - effective_step_size_before) * max_allowed_change) {

    float max_change = ((max_effective_step_size - effective_step_size_before) * max_allowed_change);
    float actual_change = effective_step_size_after - effective_step_size_before;
    float temp_meta_lr_change = max_change / actual_change;
    for (int inner_c = 0; inner_c < dim; inner_c++) {
      float grad_meta = error * x[inner_c] * h[inner_c];
      this->step_sizes[inner_c] =
          old_step_sizes[inner_c] + (step_sizes[inner_c] - old_step_sizes[inner_c]) * temp_meta_lr_change;
      this->B[inner_c] = log(this->step_sizes[inner_c]);

    }
    float new_effective_step_size = 0;
    for (int inner_c = 0; inner_c < dim; inner_c++)
      new_effective_step_size += step_sizes[inner_c];
//    std::cout << "Cutting max change by a factor of " << temp_meta_lr_change << std::endl;

//    std::cout << effective_step_size_before << " " << effective_step_size_after  <<  " " << new_effective_step_size << std::endl;
  } else if (effective_step_size_before - effective_step_size_after
      > (effective_step_size_before - min_effective_step_size) * max_allowed_change) {

    float max_change = ((effective_step_size_before - min_effective_step_size) * max_allowed_change);
    float actual_change = effective_step_size_before - effective_step_size_after;
    float temp_meta_lr_change = max_change / actual_change;
    for (int inner_c = 0; inner_c < dim; inner_c++) {
      this->step_sizes[inner_c] =
          old_step_sizes[inner_c] + (step_sizes[inner_c] - old_step_sizes[inner_c]) * temp_meta_lr_change;
      this->B[inner_c] = log(this->step_sizes[inner_c]);
//      float grad_meta = error * x[inner_c] * h[inner_c];
//      this->B[inner_c] = old_B[inner_c] + (meta_step_size * temp_meta_lr_change * grad_meta);
//      this->step_sizes[inner_c] = exp(this->B[inner_c]);
    }
  }



  for (int inner_c = 0; inner_c < dim; inner_c++) {
    if (isinf(step_sizes[inner_c]) || isnan(step_sizes[inner_c])) {
      step_sizes[inner_c] = old_step_sizes[inner_c];
      this->B[inner_c] = log(this->step_sizes[inner_c]);
    }

  }

//  std::cout << "Effective step-size very small = " << effective_step_size_before << std::endl;
//  if(effective_step_size_before > 0.9 || effective_step_size_before < 1e-4){
//    std::cout << "Effective step-size very small = " << effective_step_size_before << std::endl;
//    exit(0);
//  }

  for (int c = 0; c < dim; c++) {
    float temp = (1 - step_sizes[c] * x[c] * x[c]);
    if (temp > 0)
      h[c] = h[c] * temp + step_sizes[c] * error * x[c];
    else
      h[c] = step_sizes[c] * error * x[c];
  }

  bias_step_size = 0;

  LMS::backward(x, pred, target);
}
