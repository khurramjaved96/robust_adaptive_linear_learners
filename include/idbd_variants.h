//
// Created by Khurram Javed on 2022-06-20.
//

#ifndef INCLUDE_IDBD_VARIANTS_H_
#define INCLUDE_IDBD_VARIANTS_H_

#include <vector>
#include "learner.h"


class IDBDSmoothSteps : public IDBD {
 protected:
  float effective_step_size_trace;
  float max_effective_step_size;
  float max_allowed_change;
  std::vector<float> trace_meta_grad_features;
  float trace_meta_grad_bias;
  float min_effective_step_size;
 public:
  IDBDSmoothSteps(float meta_step_size, float step_size, int d, float percentange_change);
  void backward(std::vector<float> x, float pred, float target);
};



#endif //INCLUDE_IDBD_VARIANTS_H_
