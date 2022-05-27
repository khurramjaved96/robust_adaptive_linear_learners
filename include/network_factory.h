//
// Created by Khurram Javed on 2022-05-13.
//

#ifndef INCLUDE_NETWORK_FACTORY_H_
#define INCLUDE_NETWORK_FACTORY_H_

#include "learner.h"
#include <string>
#include "experiment/Experiment.h"

class NetworkFactory {
 public:
  static Learner *get_learner(Experiment *my_experiment) {
    Learner *network;
    int FEATURES = my_experiment->get_int_param("features");
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
      network = new IDBD(my_experiment->get_float_param("meta_step_size"),
                         my_experiment->get_float_param("step_size"),
                         FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "nidbd1")
      network = new NIDBD1(my_experiment->get_float_param("meta_step_size"),
                           my_experiment->get_float_param("step_size"),
                           FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "nidbd2")
      network = new NIDBD2(my_experiment->get_float_param("meta_step_size"),
                           my_experiment->get_float_param("step_size"),
                           FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "idbdbest")
      network = new IDBDBest(my_experiment->get_float_param("meta_step_size"),
                             my_experiment->get_float_param("step_size"),
                             FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "idbdbesty")
      network = new IDBDBestYNorm(my_experiment->get_float_param("meta_step_size"),
                                  my_experiment->get_float_param("step_size"),
                                  FEATURES);
    else if (my_experiment->get_string_param("algorithm") == "idbdnorm")
      network = new IDBDNorm(my_experiment->get_float_param("meta_step_size"),
                             my_experiment->get_float_param("step_size"),
                             FEATURES);
    return network;
  }

  NetworkFactory();
};

#endif //INCLUDE_NETWORK_FACTORY_H_
