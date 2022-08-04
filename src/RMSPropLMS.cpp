#include "../include/learner.h"
#include <math.h>
#include <iostream>
#include <tgmath.h>

RMSPropLMS::RMSPropLMS(float step_size, int d, float beta, float epsilon) : LMS(step_size, d) {
    this->beta = beta;
    this->epsilon = epsilon;
    this->v_bias = 0;
    for (int i = 0; i < dim; i++) {
        this->v.push_back(0);
    }
}

void RMSPropLMS::update_RMSProp_statistics() {
    for (int i = 0; i < dim; i++) {
        this->v[i] = this->v[i] * this->beta + (1 - this->beta) * this->gradients[i] * this->gradients[i];
    }
    this->v_bias = this->v_bias * this->beta + (1 - this->beta) * this->bias_gradient;
}

void RMSPropLMS::update_parameters() {
    this->update_RMSProp_statistics();
    for (int i = 0; i < dim; i++) 
        weights[i] -= this->step_sizes[i] * gradients[i] / (sqrt(this->v[i]) + this->epsilon);
    
    bias_weight -= this->bias_step_size * bias_gradient / (sqrt(this->v_bias) + this->epsilon);
}











