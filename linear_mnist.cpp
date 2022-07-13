#include <iostream>
#include "include/utils.h"
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
#include <string>
#include "include/utils.h"
#include "include/environments/mnist_tracking.h"
#include "include/learner.h"
#include "include/network_factory.h"


int main(int argc, char *argv[]) {
  Experiment *my_experiment = new ExperimentJSON(argc, argv);

  int T = my_experiment->get_int_param("steps");
  int no_of_features = my_experiment->get_int_param("features");

//  Initialize database tables
  Metric error_metric = Metric(my_experiment->database_name, "error_table",
                               std::vector<std::string>{"run", "step", "seed",
                                                        "mean_sum_of_errors", "asymptotic_error", "accuracy"},
                               std::vector<std::string>{"int", "int", "int", "real", "real", "real"},
                               std::vector<std::string>{"run", "step", "seed"});

//  Repeat experiment for seed number of times
  for (int seed = 0; seed < my_experiment->get_int_param("seeds"); seed++) {

    MNISTTracking *env;
    env = new MNISTTracking(no_of_features,
                                 seed);

    float sum_of_error = 0;
    float last_20k_steps_error = 0;
//     Learner *network = NetworkFactory::get_learner(my_experiment);
    std::vector<Learner*> networks;
//     std::cout << "aaaaaaa" << std::endl;
      for (int i = 0; i < 10; i++) {
            networks.push_back(NetworkFactory::get_learner(my_experiment));
      }
      // std::cout << "bbbbb" << std::endl;
      float accuracy_estimate = 0.1;
      for (int step = 0; step < T; step++) {
            // load current image sample and its targets (0 ~ 9 prediction independently)
      //      Get next sample from the world
            auto x = env->step();
            int max_index = -1;
            float max_logit = -1000;
            
            for (int i = 0; i < 10; i++) {
      //      Get target/label given by the underlying target function after the agent has made the prediction
                  float origin_target = env->get_y();
                  float target = 0;
                  if (origin_target == i)
                        target = 1;

                  Learner *network = networks[i];
                  

            //      Make a prediction
                  float pred = network->forward(x);
                  if(pred > max_logit){
                        max_logit = pred;
                        max_index = i;
                  }
                 
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
            // if (T - step < 20000) {
            // int step_counter = 20000 - (T - step);
            // last_20k_steps_error = 1.0 / (step_counter)
            //       * (network->distance_to_target_weights(env->get_target_weights())
            //       + (step_counter - 1) * last_20k_steps_error);
            // }
            }
            if(max_index == env->get_y()){
                  accuracy_estimate  = accuracy_estimate*0.9999 + 0.0001;
            }
            else{
                  accuracy_estimate  = accuracy_estimate*0.9999;
            }
            std::cout << "accuracy:" << accuracy_estimate << std::endl;
    }

//    Push results in the database
    std::vector<std::string> cur_error;
    cur_error.push_back(std::to_string(my_experiment->get_int_param("run")));
    cur_error.push_back(std::to_string(T));
    cur_error.push_back(std::to_string(seed));
    cur_error.push_back(std::to_string(sum_of_error / T));
    cur_error.push_back(std::to_string(0));
    cur_error.push_back(std::to_string(accuracy_estimate));
    error_metric.record_value(cur_error);
  }

  error_metric.commit_values();
  std::cout << "Done\n";
}
