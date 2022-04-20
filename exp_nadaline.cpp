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



int main(int argc, char *argv[]) {

  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  int STEPS = my_experiment->get_int_param("steps");
  int FEATURES = my_experiment->get_int_param("features");
  float target_noise = my_experiment->get_float_param("target_noise");
  float avg_running_error = 0;

  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector<std::string>{"run", "step", "seed", "sum_of_distance", "end_distance",
                                                        "sum_of_errors", "asymptotic_error"},
                               std::vector<std::string>{"int", "int", "int", "real", "real", "real", "real"},
                               std::vector<std::string>{"run", "step", "seed"});
  for (int seed = 0; seed < my_experiment->get_int_param("seeds"); seed++) {
    SupervisedLearning *env;
    if (my_experiment->get_string_param("input_distribution") == "normal") {
      env = new SupervisedLearningNormal(my_experiment->get_float_param("x_mean"),
                                         FEATURES,
                                         seed, target_noise);
    } else if (my_experiment->get_string_param("input_distribution") == "binary") {
      env = new SupervisedLearningBinary(my_experiment->get_float_param("p"), my_experiment->get_float_param("x_mean"),
                                         FEATURES,
                                         seed, target_noise);
    } else if (my_experiment->get_string_param("input_distribution") == "normal_capped") {
      env = new SupervisedLearningNormalCapped(my_experiment->get_float_param("x_mean"),
                                               FEATURES,
                                               seed, my_experiment->get_float_param("cap"), target_noise);
    }
    else if (my_experiment->get_string_param("input_distribution") == "binary_variable") {
      env = new SupervisedLearningVariableBinary(my_experiment->get_float_param("p_start"),
                                                 my_experiment->get_float_param("p_end"),
                                                 my_experiment->get_float_param("x_mean"),
                                                 FEATURES,
                                                 seed, target_noise);

    }
    else if (my_experiment->get_string_param("input_distribution") == "uniform") {
      env = new SupervisedLearningUniform(my_experiment->get_float_param("a"),
                                                 my_experiment->get_float_param("x_mean"),
                                                 FEATURES,
                                                 seed, target_noise);

    }
//
    float avg_target = 0;
    float last_distance = 0;
    float sum_of_squared_avg_distance = 0;
    float sum_of_error = 0;
    float last_10k_error = 0;
    float variance = 0;
    Learner *network;
    if (my_experiment->get_string_param("algorithm") == "nadaline")
      network = new Nadaline(my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "lms")
      network = new LMS(my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "normalized_lms")
      network = new NormalizedLMS(my_experiment->get_float_param("step_size"), FEATURES);
    for (int step = 0; step < STEPS; step++) {
      auto x = env->step();
//      std::cout << x[0] << std::endl;
      variance = (1.0/(step+1)) * (x[0]*x[0] + (step)*variance);
//      if(x[0] > 5){
//        std::cout << "Extreme value\n";
//        std::cout << x[0] << std::endl;
//      }
      float pred = network->forward(x);
      float target = env->get_y();
      avg_target = 1.0 / (float(step) + 1.0) * (target + float(step) * avg_target);
      float mse = (target - pred) * (target - pred);
      if(STEPS - step < 10000) {
        int step_counter = 10000 - (STEPS - step);
        last_10k_error = 1.0/(step_counter) * (network->distance_to_target_weights(env->get_target_weights()) + (step_counter - 1) * last_10k_error );
      }
      sum_of_error += mse;
      network->zero_grad();
      network->backward(x, pred, target);
      if (step > 5)
        network->update_parameters();
      last_distance = network->distance_to_target_weights(env->get_target_weights());
      sum_of_squared_avg_distance += last_distance;
//        print_vector(env->get_target_weights());
    }

//    std::cout << "Avg distance = " << sum_of_squared_avg_distance / STEPS << std::endl;
    std::vector<std::string> cur_error;
    cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
    cur_error.push_back(std::to_string(STEPS));
    cur_error.push_back(std::to_string(seed));
    cur_error.push_back(std::to_string(sum_of_squared_avg_distance / STEPS));
    cur_error.push_back(std::to_string(last_distance));
    cur_error.push_back(std::to_string(sum_of_error / STEPS));
    cur_error.push_back(std::to_string(last_10k_error));
    error_metric.record_value(cur_error);
//    std::cout << "Variance = " <<  my_experiment->get_string_param("input_distribution") << " " <<  variance << std::endl;
  }
//  std::cout << "####### ERROR IS HERE ########\n\n\n";
//  std::cout << "Algorthm = " << my_experiment->get_string_param("input_distribution") << std::endl;
//  std::cout << "running error = " << running_error << std::endl;
//  std::cout << "Avg error = " << avg_running_error << std::endl;
  error_metric.commit_values();

  std::cout << "Done\n";
}
//