cmake_minimum_required(VERSION 3.14)
project(LMS LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#add_executable(NormalizedLMS include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
#        exp_LMS.cpp src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp)
#
#add_executable(Nadaline include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
#        exp_nadaline.cpp  src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp)
#

#add_executable(Robust include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
#        robust_linear_learner.cpp  src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp include/idbd_variants.h src/idbd_variants.cpp)


# add_executable(CreditAssignment include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
#         credit_assignment_plots.cpp  src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp include/idbd_variants.h src/idbd_variants.cpp)


add_executable(Count include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        count_configs.cpp  src/environments/supervised_tracking.cpp include/environments/supervised_tracking.h include/learner.h src/learner.cpp)


add_executable(MNIST include/json.hpp include/utils.h src/utils.cpp include/experiment/Experiment.h include/experiment/Metric.h include/experiment/Database.h src/experiment/Metric.cpp src/experiment/Database.cpp src/experiment/Experiment.cpp
        linear_mnist.cpp include/environments/mnist/mnist_reader.hpp include/environments/mnist/mnist_reader_common.hpp include/environments/mnist/mnist_reader_less.hpp include/environments/mnist/mnist_utils.hpp src/environments/mnist_tracking.cpp include/environments/mnist_tracking.h include/learner.h src/learner.cpp)
##

#
#target_link_libraries(NormalizedLMS PUBLIC mariadb)
#target_link_libraries(NormalizedLMS PUBLIC tbb)

#target_link_libraries(Nadaline PUBLIC mariadb)
#target_link_libraries(Nadaline PUBLIC tbb)

# target_link_libraries(CreditAssignment PUBLIC mariadb)
# target_link_libraries(CreditAssignment PUBLIC tbb)

target_link_libraries(MNIST PUBLIC mariadb)
target_link_libraries(MNIST PUBLIC tbb)

#target_link_libraries(Robust PUBLIC mariadb)
#target_link_libraries(Robust PUBLIC tbb)

target_link_libraries(Count PUBLIC mariadb)
target_link_libraries(Count PUBLIC tbb)


project(LMS LANGUAGES CXX)

