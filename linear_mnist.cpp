#include <iostream>
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include <string>
#include "include/utils.h"
#include "include/environments/supervised_tracking.h"
#include "include/learner.h"
#include "include/network_factory.h"

std::vector<std::vector<float>> change_mnist_targets(std::vector<std::vector<float>> targets, int total_data_points) {
      for (int i = 0; i < total_data_points; i++) {
            targets[i][0] = rand() % 10;
      }
      return targets;
}


int main(int argc, char *argv[]) {
  Experiment *my_experiment = new ExperimentJSON(argc, argv);
  
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
                                                              mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("mnist_data/");

  int total_data_points = 60000;
  int total_test_points = 10000;
  
  std::vector<std::vector<float>> images;
  std::vector<std::vector<float>> targets;
  std::vector<std::vector<float>> images_test;
  std::vector<std::vector<float>> targets_test;


  for(int counter = 0; counter < total_data_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.training_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    y_temp.push_back(float(unsigned(dataset.training_labels[counter])));
    images.push_back(x_temp);
    targets.push_back(y_temp);
  }

  for(int counter = 0; counter < total_test_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.test_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    std::vector<float> y_temp;
    y_temp.push_back(float(unsigned(dataset.test_labels[counter])));
    images_test.push_back(x_temp);
    targets_test.push_back(y_temp);
  }


  int T = my_experiment->get_int_param("steps");
  int no_of_features = my_experiment->get_int_param("features");
  float noise_in_the_target = my_experiment->get_float_param("target_noise");

//  Initialize database tables
  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector<std::string>{"run", "step", "seed",
                                                        "mean_sum_of_errors", "asymptotic_error"},
                               std::vector<std::string>{"int", "int", "int", "real", "real"},
                               std::vector<std::string>{"run", "step", "seed"});

//  Repeat experiment for seed number of times
  for (int seed = 0; seed < my_experiment->get_int_param("seeds"); seed++) {
//    Initialize the environment
    SupervisedTracking *env;
//     env = new MNIST_Tracking(   images, targets
//                                  no_of_features,
//                                  seed,
//                                  my_experiment->get_float_param("target_noise"));

    float sum_of_error = 0;
    float last_20k_steps_error = 0;
    Learner *network = NetworkFactory::get_learner(my_experiment);

    for (int step = 0; step < T; step++) {
      // load current image sample and its targets (0 ~ 9 prediction independently)
      // auto x = env->step();
      auto x = images[step % total_data_points];
      float target = targets[step % total_data_points][0];

      if (step % 200 == 0):
      // randomly change target mapping each 100 time step
            targets = change_mnist_targets();

      for (i = 0; i < 10; i++) {
            float sub_target = 0;
            if (i == target)
                  sub_target = 1;

      //      Make a prediction
            float pred = network->forward(x);
      //      Get target/label given by the underlying target function after the agent has made the prediction
            
      //      Compute the squared error
            float squared_error = (target - pred) * (target - pred);
      //      Set the gradient accumulation vector to zero
            network->zero_grad();
      //      Add gradient of the 1/2 (target - pred)^2 w.r.t the learnable parameters to the gradient accumulation vector
            network->backward(x, pred, target);
      //      Update the parameters using the stored gradients
            network->update_parameters();
      //      Update the sum of errors so far
            sum_of_error += squared_error;
      //      Update asymptotic error (Error on last 20k steps).
            if (T - step < 20000) {
            int step_counter = 20000 - (T - step);
            last_20k_steps_error = 1.0 / (step_counter)
                  * (network->distance_to_target_weights(env->get_target_weights())
                  + (step_counter - 1) * last_20k_steps_error);
            }
      }
    }

//    Push results in the database
    std::vector<std::string> cur_error;
    cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
    cur_error.push_back(std::to_string(T));
    cur_error.push_back(std::to_string(seed));
    cur_error.push_back(std::to_string(sum_of_error / T));
    cur_error.push_back(std::to_string(last_20k_steps_error));
    error_metric.record_value(cur_error);
  }

  error_metric.commit_values();
  std::cout << "Done\n";
}
