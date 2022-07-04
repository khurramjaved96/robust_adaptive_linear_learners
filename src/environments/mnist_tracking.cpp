#include "../../include/environments/mnist_tracking.h"
#include <iostream>
#include "../../include/environments/mnist/mnist_reader.hpp"
#include "../../include/environments/mnist/mnist_utils.hpp"

MNISTTracking::MNISTTracking(int dimensions,
                                       int seed) : dimension(dimensions),
                                                             time(0) {

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                                                              mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("mnist_data/");


  for(int counter = 0; counter < total_data_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.training_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    // std::cout << size(x_temp) << std::endl;
    std::vector<float> y_temp;
    y_temp.push_back(float(unsigned(dataset.training_labels[counter])));
    images.push_back(x_temp);
    targets.push_back(y_temp);
  } 

  for (int i = 0; i < 10; i++) {
    std::vector<float> temp;
    for (int j = 0; j < 2; j++) 
      temp.push_back(i);
    this->mappings.push_back(temp);
  }
}

void MNISTTracking::change_target() {

  int index1 = rand() % 10;
  int index2 = rand() % 10;

  float temp = this->mappings[index1][1];
  this->mappings[index1][1] = this->mappings[index2][1];
  this->mappings[index2][1] = temp;


}

float MNISTTracking::map(float target) {
    return this->mappings[target][1];
}

std::vector<float> MNISTTracking::step() {
  time++;
  this->current_index = rand() % this->total_data_points;
  if (time % 500 == 0)
    change_target();

  return this->images[current_index];
}

float MNISTTracking::get_y() {
  return map(this->targets[current_index][0]);
}
