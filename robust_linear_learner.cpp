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
    SupervisedTracking *env;
    env = new SupervisedTracking(my_experiment->get_float_param("input_mean"),
                                 my_experiment->get_float_param("input_std"),
                                 my_experiment->get_float_param("target_mean"),
                                 my_experiment->get_float_param("target_std"),
                                 FEATURES,
                                 seed,
                                 my_experiment->get_float_param("target_noise"));

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
    else if (my_experiment->get_string_param("algorithm") == "lms_normalized_step_size")
      network = new LMSNormalizedStepSize(my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "lms_normalized_step_size_and_input")
      network = new LMSNormalizedInputsAndStepSizes(my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "normalized_lms")
      network = new NormalizedLMS(my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "adam")
      network = new AdamLMS(my_experiment->get_float_param("step_size"),
                            FEATURES,
                            my_experiment->get_float_param("b1"),
                            my_experiment->get_float_param("b2"),
                            my_experiment->get_float_param("epsilon"));
    else if (my_experiment->get_string_param("algorithm") == "idbd")
      network = new IDBD(my_experiment->get_float_param("meta_step_size"), my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "nidbd1")
      network = new NIDBD1(my_experiment->get_float_param("meta_step_size"), my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "nidbd2")
      network = new NIDBD2(my_experiment->get_float_param("meta_step_size"), my_experiment->get_float_param("step_size"), FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "idbdbest")
      network = new IDBDBest(my_experiment->get_float_param("meta_step_size"), my_experiment->get_float_param("step_size"), FEATURES);

    for (int step = 0; step < STEPS; step++) {
      auto x = env->step();
      float pred = network->forward(x);
      float target = env->get_y();
//      std::cout << "target = " << target << std::endl;
//      std::cout << "pred = " << pred << std::endl;
      avg_target = 1.0 / (float(step) + 1.0) * (target + float(step) * avg_target);
      float mse = (target - pred) * (target - pred);
      if (STEPS - step < 20000) {
        int step_counter = 20000 - (STEPS - step);
        last_10k_error = 1.0 / (step_counter)
            * (network->distance_to_target_weights(env->get_target_weights()) + (step_counter - 1) * last_10k_error);
      }
//      std::cout << "last 20 k = " << last_10k_error << std::endl;
      sum_of_error += mse;
      network->zero_grad();
      network->backward(x, pred, target);
      if(step > 3000)
        network->update_parameters();
      last_distance = network->distance_to_target_weights(env->get_target_weights());
      sum_of_squared_avg_distance += last_distance;
    }

    std::vector<std::string> cur_error;
    cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
    cur_error.push_back(std::to_string(STEPS));
    cur_error.push_back(std::to_string(seed));
    cur_error.push_back(std::to_string(sum_of_squared_avg_distance / STEPS));
    cur_error.push_back(std::to_string(last_distance));
    cur_error.push_back(std::to_string(sum_of_error / STEPS));
    cur_error.push_back(std::to_string(last_10k_error));
    error_metric.record_value(cur_error);
//    LMSNormalizedInputsAndStepSizes* ptr = dynamic_cast<LMSNormalizedInputsAndStepSizes*>(network);
//    std::cout << "Mean\n";
//    print_vector(ptr->input_normalization_mean);
//    std::cout << "Variance\n";
//    print_vector(ptr->input_normalization_std);
  }

  error_metric.commit_values();

  std::cout << "Done\n";
}
