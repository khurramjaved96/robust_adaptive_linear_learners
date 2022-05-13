//
// Created by Khurram Javed on 2022-02-02.
//

#ifndef INCLUDE_LEARNER_H_
#define INCLUDE_LEARNER_H_
#include <vector>


//class TheoryLearner{
//  std::vector<float> weights;
//  int dim;
//  float h_1;
//  float b_1_tilde;
//  float B_1_tilde;
//  float G_0;
//  float D_1;
//  std::vector<float> theta_tilde;
//  float epsilon;
//  std::vector<std::vector<float>> weights_eta;
//};

class Learner {
 protected:
  float bias_weight;
  float counter;
  std::vector<float> weights;
  int dim;
//  float step_size_normalization;
  std::vector<float> step_sizes;
  std::vector<float> gradients;
  float bias_gradient;
 public:
  Learner(float step_size, int d);
  float get_dot_product(std::vector<float> my_vec);
  virtual float forward(std::vector<float> x) = 0;
  virtual void backward(std::vector<float> x, float pred, float target) = 0;
  virtual void update_parameters() = 0;
  virtual void zero_grad();
  virtual float distance_to_target_weights(std::vector<float> target_weights);
  std::vector<float> get_weights();
};

class LMS : public Learner {
 protected:
  float bias_step_size;
  float target_test_mean;

 public:
  virtual float forward(std::vector<float> x);
  virtual void backward(std::vector<float> x, float pred, float target);
  virtual void update_parameters();
  LMS(float step_size, int d);
};

class LMSNormalizedStepSize : public LMS {
 protected:
  float step_size_normalization;
 public:
  virtual float forward(std::vector<float> x);
  virtual void update_parameters();
  LMSNormalizedStepSize(float step_size, int d);
};

class NormalizedLMS : public LMS {
  float cur_dot_product;
 public:
  virtual void update_parameters();
  virtual float forward(std::vector<float> x);
  NormalizedLMS(float step_size, int d);
};

class LMSNormalizedInputsAndStepSizes : public LMSNormalizedStepSize {
 protected:
  std::vector<float> normalize_x(std::vector<float> x);

  void update_normalization_estimates(std::vector<float> x);
 public:
  std::vector<float> input_normalization_mean;
  std::vector<float> input_normalization_std;
  virtual float forward(std::vector<float> x);
  virtual void backward(std::vector<float> x, float pred, float target);
  LMSNormalizedInputsAndStepSizes(float step_size, int d);
};

class Nadaline : public Learner {
 protected:
  std::vector<float> normalize_x(std::vector<float> x);
  std::vector<float> input_normalization_mean;
  std::vector<float> input_normalization_std;
  float target_test_mean;
  float target_test_std;
  void update_normalization_estimates(std::vector<float> x);
 public:
  float forward(std::vector<float> x);
  void backward(std::vector<float> x, float pred, float target);
  void update_parameters();
  Nadaline(float step_size, int d);
};

class AdamLMS : public LMS {
 protected:
  std::vector<float> m1;
  std::vector<float> m2;
  float b1;
  float b2;
  float epsilon;
  float m1_bias;
  float m2_bias;
  void update_adam_statistics();
 public:
  void update_parameters();
  AdamLMS(float step_size, int d, float b1, float b2, float epsilon);
};

class NormalizedIDBD : public LMSNormalizedInputsAndStepSizes {
 protected:
  std::vector<float> h;
  std::vector<float> B;
  std::vector<float> step_size_gradients;
  float meta_step_size;
 public:
  NormalizedIDBD(float meta_step_size, float step_size, int d);
  void backward(std::vector<float> x, float pred, float target);
};

class IDBD : public LMS {
 protected:
  std::vector<float> h;
  std::vector<float> B;
  std::vector<float> step_size_gradients;
  float h_bias;
  float B_bias;
  float step_size_gradient_bias;
  float meta_step_size;
 public:
  IDBD(float meta_step_size, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);
};

class IDBDBest : public LMSNormalizedInputsAndStepSizes{
 protected:
  std::vector<float> std_delta;
  std::vector<float> mean_delta;
  float std_bias_delta;
  float mean_bias_delta;
  std::vector<float> h;
  std::vector<float> B;
  std::vector<float> step_size_gradients;
  float h_bias;
  float B_bias;
  float step_size_gradient_bias;
  float meta_step_size;
 public:
  IDBDBest(float meta_step_size, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);

};

class NIDBD1 : public IDBD {
 protected:
  float std_delta;
  float mean_delta;
 public:
  NIDBD1(float meta_step_size, float step_size, int d);
  void backward(std::vector<float> x, float pred, float target);
};


class NIDBD2 : public IDBD {
 protected:
  std::vector<float> std_delta;
  std::vector<float> mean_delta;
  float std_bias_delta;
  float mean_bias_delta;
 public:
  NIDBD2(float meta_step_size, float step_size, int d);
  void backward(std::vector<float> x, float pred, float target);
};



//
//class LMS_Input_Normalization : public LMS{
// protected:
//  void update_normalization_estimates(std::vector<float> x);
//  std::vector<float> normalize_x(std::vector<float> x);
//  std::vector<float> input_normalization_mean;
//  std::vector<float> input_normalization_std;
//  std::vector<float> target_normalization_mean;
//  std::vector<float> target_normalization_std;
// public:
//  LMS_Input_Normalization(float step_size, int d);
//  float forward(std::vector<float> x);
//  void backward(std::vector<float> x, float pred, float target);
//  std::vector<float> get_input_mean();
//  std::vector<float> get_input_std();
//};
//
//
//class LMS_Input_target_normalization : public LMS_Input_Normalization{
// protected:
//  float target_mean;
//  float target_std;
// public:
//  LMS_Input_target_normalization(float step_size, int d);
//  float forward(std::vector<float> x);
//  void backward(std::vector<float> x, float pred, float target);
//  void update_target_statistics(float target);
//};
//
//class IDBD : public LMS_Input_target_normalization{
// protected:
//  std::vector<float> h;
//  std::vector<float> B;
//  std::vector<float> step_size_graidents;
//  float meta_step_size;
// public:
//  IDBD(float meta_step_size, int d);
//  float forward(std::vector<float> x);
//  void backward(std::vector<float> x, float pred, float target);
//  void update_step_size();
//  void update_parameters();
//};
//
// class IDBDNormalized : public LMS_Input_Normalization
// {
//  protected:
//   std::vector<float> h;
//   std::vector<float> B;
//   std::vector<float> step_size_graidents;
//   float meta_step_size;
//   float target_mean;
//   float target_std;
//  public:
//   IDBDNormalized(float meta_step_size, int d);
//   float forward(std::vector<float> x);
//   void backward(std::vector<float> x, float pred, float target);
//   void update_step_size();
//   void update_parameters();
//   void update_target_statistics(float target);
// };


#endif //INCLUDE_LEARNER_H_
