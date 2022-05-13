//
// Created by Khurram Javed on 2022-01-08.
//

#include <iostream>
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include <string>
#include "include/utils.h"
#include "include/environments/supervised_tracking.h"
#include "include/learner.h"
#include "include/network_factory.h"

int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

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
    env = new SupervisedTracking(my_experiment->get_float_param("input_mean"),
                                 my_experiment->get_float_param("input_std"),
                                 my_experiment->get_float_param("target_mean"),
                                 my_experiment->get_float_param("target_std"),
                                 no_of_features,
                                 seed,
                                 my_experiment->get_float_param("target_noise"));

    float sum_of_error = 0;
    float last_20k_steps_error = 0;
    Learner *network = NetworkFactory::get_learner(my_experiment);

    for (int step = 0; step < T; step++) {
//      Get next sample from the world
      auto x = env->step();
//      Make a prediction
      float pred = network->forward(x);
//      Get target/label given by the underlying target function after the agent has made the prediction
      float target = env->get_y();
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
