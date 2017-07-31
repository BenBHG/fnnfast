#include "fnnfast.h"
#include <string.h>
#include <stdio.h>

size_t fnnfast_size(struct fnnfast_data *data) {
	size_t bytes = 0;
	// hidden neurons
	bytes += data->num_hidden;
	// delta input hidden
	bytes += (data->num_input + 1) * data->num_hidden;
	// delta hidden output
	bytes += (data->num_hidden + 1) * data->num_output;
	// input weights
	bytes += (data->num_input + 1) * data->num_hidden;
	// hidden weights
	bytes += (data->num_hidden + 1) * data->num_output;
	bytes *= sizeof(neuron);
	bytes += sizeof(struct fnnfast_data);
	return bytes;
}

struct fnnfast_data * fnnfast_new(size_t num_input, size_t num_hidden, size_t num_output) {
	struct fnnfast_data temp;
	temp.num_input = num_input;
	temp.num_hidden = num_hidden;
	temp.num_output = num_output;
	size_t sz = fnnfast_size(&temp);
	struct fnnfast_data *data = malloc(sz);
	data->num_input = num_input;
	data->num_hidden = num_hidden;
	data->num_output = num_output;
	fnnfast_fix_pointers(data);
	fnnfast_zero(data);
	return data;
}

void fnnfast_delete(struct fnnfast_data *data) {
	free(data);
}

void fnnfast_fix_pointers(struct fnnfast_data *data) {
	data->p_hidden_neurons = (neuron *)((char *)data + sizeof(struct fnnfast_data));
	data->p_delta_input_hidden = data->p_hidden_neurons + data->num_hidden;
	data->p_delta_hidden_output = data->p_delta_input_hidden + (data->num_input + 1) * data->num_hidden;
	data->p_input_weights = data->p_delta_hidden_output + (data->num_hidden + 1) * data->num_output;
	data->p_hidden_weights = data->p_input_weights + (data->num_input + 1) * data->num_hidden;
}

void fnnfast_zero(struct fnnfast_data *data) {
	neuron *end = data->p_hidden_weights + (data->num_hidden + 1) * data->num_output;
	neuron *cur = data->p_hidden_neurons;
	while (cur != end) {
		*cur = NEURON_ZERO;
		++cur;
	}
}

void fnnfast_randomize(struct fnnfast_data *data, unsigned int seed) {
	neuron *end = data->p_hidden_weights + (data->num_hidden + 1) * data->num_output;
	neuron *cur = data->p_input_weights;
	srand(seed);
	while (cur != end) {
		*cur = (neuron)rand() / (neuron)(RAND_MAX) - NEURON_ONE / NEURON_TWO;
		++cur;
	}
}

static inline neuron fnnfast_activate(neuron n) {
	return NEURON_ONE / (NEURON_ONE + exp_neuron(-n));
}

void fnnfast_feedforward(struct fnnfast_data *data, neuron *input, neuron *output) {
	for (size_t i = 0; i < data->num_hidden; ++i) {
		data->p_hidden_neurons[i] = NEURON_ZERO;
		for (size_t j = 0; j < data->num_input; ++j)
			data->p_hidden_neurons[i] += input[j] * data->p_input_weights[j * data->num_hidden + i];
		data->p_hidden_neurons[i] += -NEURON_ONE * data->p_input_weights[data->num_input * data->num_hidden + i];
		data->p_hidden_neurons[i] = fnnfast_activate(data->p_hidden_neurons[i]);
	}
	for (size_t i = 0; i < data->num_output; ++i) {
		output[i] = NEURON_ZERO;
		for (size_t j = 0; j < data->num_hidden; ++j)
			output[i] += data->p_hidden_neurons[j] * data->p_hidden_weights[j * data->num_output + i];
		output[i] += -NEURON_ONE * data->p_hidden_weights[data->num_hidden * data->num_output + i];
		output[i] = fnnfast_activate(output[i]);
	}
}

neuron fnnfast_mean_squared_deviation(struct fnnfast_data *data, neuron **input_set, neuron **output_set, size_t num) {
	neuron sq_deviation = NEURON_ZERO;
	neuron *output = malloc(data->num_output * sizeof(neuron));
	for (size_t i = 0; i < num; ++i) {
		fnnfast_feedforward(data, input_set[i], output);
		for (size_t j = 0; j < data->num_output; ++j) {
			sq_deviation += pow_neuron(output[j] - output_set[i][j], NEURON_TWO);
		}
	}
	free(output);
	return sq_deviation / (data->num_output * num);
}

static inline neuron fnnfast_output_error_gradient(neuron target, neuron output) {
	return output * (NEURON_ONE - output) * (target - output);
}

static inline neuron fnnfast_hidden_error_gradient(struct fnnfast_data *data, size_t h, neuron *target, neuron *output) {
	neuron sum = NEURON_ZERO;
	for (size_t i = 0; i < data->num_output; ++i)
		sum += data->p_hidden_weights[h * data->num_output + i] * fnnfast_output_error_gradient(target[i], output[i]);
	return data->p_hidden_neurons[h] * (NEURON_ONE - data->p_hidden_neurons[h]) * sum;
}

static void fnnfast_backpropogate(struct fnnfast_data *data, neuron *input, neuron *training_output, neuron *target_output, neuron rate, neuron momentum) {
	for (size_t i = 0; i < data->num_output; ++i) {
		neuron correction = fnnfast_output_error_gradient(target_output[i], training_output[i]);
		for (size_t j = 0; j < data->num_hidden; ++j) {
			data->p_delta_hidden_output[j * data->num_output + i] = rate * data->p_hidden_neurons[j] * correction + momentum * data->p_delta_hidden_output[j * data->num_output + i];
			data->p_hidden_weights[j * data->num_output + i] += data->p_delta_hidden_output[j * data->num_output + i];
		}
		data->p_delta_hidden_output[data->num_hidden * data->num_output + i] = rate * (-NEURON_ONE) * correction + momentum * data->p_delta_hidden_output[data->num_hidden * data->num_output + i];
		data->p_hidden_weights[data->num_hidden * data->num_output + i] += data->p_delta_hidden_output[data->num_hidden * data->num_output + i];
	}
	for (size_t i = 0; i < data->num_hidden; ++i) {
		neuron correction = fnnfast_hidden_error_gradient(data, i, target_output, training_output);
		for (size_t j = 0; j < data->num_input; ++j) {
			data->p_delta_input_hidden[j * data->num_hidden + i] = rate * input[j] * correction + momentum * data->p_delta_input_hidden[j * data->num_hidden + i];
			data->p_input_weights[j * data->num_hidden + i] += data->p_delta_input_hidden[j * data->num_hidden + i];
		}
		data->p_delta_input_hidden[data->num_input * data->num_hidden + i] = rate * (-NEURON_ONE) * correction + momentum * data->p_delta_input_hidden[data->num_input * data->num_hidden + i];
		data->p_input_weights[data->num_input * data->num_hidden + i] += data->p_delta_input_hidden[data->num_input * data->num_hidden + i];
	}
}

void fnnfast_train(struct fnnfast_data *data, neuron *input, neuron *output, neuron rate, neuron momentum, neuron *training_output_buf) {
	fnnfast_feedforward(data, input, training_output_buf);
	fnnfast_backpropogate(data, input, training_output_buf, output, rate, momentum);
}
