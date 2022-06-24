//
// Created by Khurram Javed on 2022-02-02.
//

#ifndef INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
#define INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
#include <vector>
#include <random>

class SupervisedTracking {
 protected:
  std::mt19937 mt;
  std::bernoulli_distribution input_sampler;
  std::uniform_real_distribution<float> target_noise_sampler;
  std::uniform_int_distribution<int> weight_change_index_sampler;
  float input_mean;
  float input_std;
  float target_mean;
  long long int time;
  std::vector<float> base_x;
  float target_std;
  int dimension;
  std::vector<float> target_weights;
  void change_target();
 public:
  std::vector<float> get_target_weights();
  SupervisedTracking(float input_mean,
                     float input_std,
                     float target_weights_mean,
                     float target_weights_std,
                     int dimensions,
                     int seed,
                     float target_noise);
  virtual std::vector<float> step();

  float get_y();
  std::vector<float> generate_random_x();

};

class SupervisedTrackingFeaturewiseNonstationarity : public SupervisedTracking{
 public:
  std::discrete_distribution<int> discrete_distribution;
  std::vector<int> prob_of_weight_flip;
  void change_weights_based_on_probability();
  std::vector<float> step();
  SupervisedTrackingFeaturewiseNonstationarity(float input_mean,
                                               float input_std,
                                               float target_weights_mean,
                                               float target_weights_std,
                                               int dimensions,
                                               int seed,
                                               float target_noise,
                                               std::vector<int> prob_of_weight_flip_init);
};

class SupervisedLearning {
 protected:
  std::mt19937 mt;
  std::uniform_real_distribution<float> target_noise_sampler;
  std::uniform_int_distribution<int> weight_change_index_sampler;
  long long int time;
  std::vector<float> base_x;
  int dimension;
  std::vector<float> target_weights;
 public:
  void change_target_weights();
  std::vector<float> get_target_weights();
  SupervisedLearning(int dimensions, int seed, float target_noise);
  virtual std::vector<float> step() = 0;
  virtual std::vector<float> generate_random_x() = 0;
  virtual float get_y();
};

class SupervisedLearningNormal : public SupervisedLearning {
 protected:
  std::normal_distribution<float> input_sampler;
  float input_mean;
 public:
  SupervisedLearningNormal(float input_mean, int dimensions, int seed, float target_noise );
  virtual std::vector<float> step();
  virtual std::vector<float> generate_random_x();
};

class SupervisedLearningBinary : public SupervisedLearning {
 protected:
  float p_val;
  float k;
  float input_mean;
  std::bernoulli_distribution inp_distribution;
 public:
  SupervisedLearningBinary(float p, float input_mean, int dimensions, int seed, float target_noise);
  virtual std::vector<float> step();
  std::vector<float> generate_random_x();
};

class SupervisedLearningUniform : public SupervisedLearning {
 protected:
  float p_val;
  float k;
  float input_mean;
  std::uniform_real_distribution<float> inp_distribution;
 public:
  SupervisedLearningUniform(float a, float input_mean, int dimensions, int seed, float target_noise);
  virtual std::vector<float> step();
  std::vector<float> generate_random_x();
};

class SupervisedLearningVariableBinary : public SupervisedLearning {
 protected:
  std::vector<float> p_val_vector;
  std::vector<float> k_vector;
  float input_mean;
  std::vector<std::bernoulli_distribution> inp_distribution_vec;
 public:
  SupervisedLearningVariableBinary(float p_start, float increment, float input_mean, int dimensions, int seed, float target_noise);
  virtual std::vector<float> step();
  std::vector<float> generate_random_x();
};

class SupervisedLearningNormalCapped : public SupervisedLearningNormal {
 protected:
  float cap;
 public:
  SupervisedLearningNormalCapped(float input_mean, int dimensions, int seed, float cap, float target_noise);
  virtual std::vector<float> generate_random_x();
};

#endif //INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
