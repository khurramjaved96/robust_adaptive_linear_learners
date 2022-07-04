#ifndef INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
#define INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_
#include <vector>
#include <random>

class MNISTTracking {
 protected:
  long long int time;
  int dimension;
  std::vector<float> target_weights;
  std::vector<std::vector<float>> mappings;
  void change_target();
  int current_index;
 public:
  std::vector<float> get_target_weights();
  MNISTTracking(int dimensions,
                     int seed);
  virtual std::vector<float> step();
  float map(float target);

  float get_y();


  int total_data_points = 60000;
  int total_test_points = 10000;
  
  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;
  std::vector<std::vector<float>> images_test;
  std::vector<std::vector<float>> targets_test;


};
#endif //INCLUDE_ENVIRONMENTS_SUPERVISED_TRACKING_H_