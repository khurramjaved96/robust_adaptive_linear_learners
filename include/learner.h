//
// Created by Khurram Javed on 2022-02-02.
//

#ifndef INCLUDE_LEARNER_H_
#define INCLUDE_LEARNER_H_
#include <vector>

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
  std::vector<float> get_step_sizes();
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

  std::vector<float> step_size_gradients;
  float h_bias;
  float B_bias;
  float step_size_gradient_bias;
  float meta_step_size;
 public:
  std::vector<float> get_step_sizes();
  std::vector<float> B;
  IDBD(float meta_step_size, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);
};

class IDBDBetaNorm : public LMS {
 protected:
  std::vector<float> h;
  std::vector<float> B;

  std::vector<float> step_size_gradients;
  float h_bias;
  float B_bias;
  float step_size_gradient_bias;
  float meta_step_size;
  std::vector<float> std_delta;
  std::vector<float> mean_delta;
  float std_bias_delta;
  float mean_bias_delta;
 public:
  IDBDBetaNorm(float meta_step_size, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);
};

class IDBDBest : public LMSNormalizedInputsAndStepSizes {
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
  void print_information(std::vector<float> x, float pred, float target);
  IDBDBest(float meta_step_size, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);

};

class IDBDNorm : public LMSNormalizedInputsAndStepSizes {
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
  void print_information(std::vector<float> x, float pred, float target);
  IDBDNorm(float meta_step_size, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);
};


class IDBDBestYNorm : public LMSNormalizedInputsAndStepSizes {
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
  IDBDBestYNorm(float meta_step_size, float step_size, int d);
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

class RMSPropLMS: public LMS {
  protected:
    std::vector<float> v;
    float beta;
    float epsilon;
    float v_bias;
    void update_RMSProp_statistics();
  public:
    void update_parameters();
    RMSPropLMS(float step_size, int d, float beta, float epsilon);
};

class AdagradLMS: public LMS {
  protected:
    std::vector<float> v;
    float v_bias;
    float step_size_decay;
    float epsilon;
    void update_Adagrad_statistics();
  public:
    void update_parameters();
    AdagradLMS(float step_size, int d, float step_size_decay, float epsilon);
};

class AdadeltaLMS: public LMS {
  protected:
    std::vector<float> v;
    std::vector<float> u;
    std::vector<float> delta_x;
    float v_bias;
    float u_bias;
    float delta_x_bias;
    float decay;
    float epsilon;
    void update_Adadelta_statistics();
  public:
    void update_parameters();
    AdadeltaLMS(float step_size, int d, float decay, float epsilon);
};

class SigIDBD : public LMS {
 protected:
  float h_z;
  std::vector<float> h_iz;
  std::vector<float> h;
  float z;
  float theta_z;
  std::vector<float> step_size_gradients;
  float bias_h_z;
  float bias_h_iz;
  float h_bias;
  float B_bias;
  float bias_mu;
  float step_size_gradient_bias;
  float meta_step_size;
  float bias_z;
  float m;
  float mu;
 public:
  std::vector<float> get_step_sizes();
  std::vector<float> B;
  SigIDBD(float meta_step_size, float theta_z, float step_size, int d);
  virtual void backward(std::vector<float> x, float pred, float target);
  float get_mu();
  void update_parameters();
};


#endif //INCLUDE_LEARNER_H_
