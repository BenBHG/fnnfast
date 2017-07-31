#ifndef __fnnfast_h__
#define __fnnfast_h__

#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifdef __x86_64__
typedef double neuron;
#define exp_neuron exp
#define pow_neuron pow
#define fabs_neuron fabs
#define NEURON_TWO 2.0
#define NEURON_ONE 1.0
#define NEURON_ZERO 0.0
#else
typedef float neuron;
#define exp_neuron expf
#define pow_neuron powf
#define fabs_neuron fabsf
#define NEURON_TWO 2.0f
#define NEURON_ONE 1.0f
#define NEURON_ZERO 0.0f
#endif

struct fnnfast_data {
	size_t num_input;
	size_t num_hidden;
	size_t num_output;
	neuron *p_hidden_neurons;
	neuron *p_delta_input_hidden;
	neuron *p_delta_hidden_output;
	neuron *p_input_weights;
	neuron *p_hidden_weights;
};

/**
 * Calculates the size in bytes that a fnnfast_data instance should be or is.
 * 
 * @param data      			pointer to fnnfast_data structure
 * @return size
 */
size_t fnnfast_size(struct fnnfast_data *data);

/**
 * Allocates a new fnnfast_data instance.
 * 
 * @param num_input 			number of inputs
 * @param num_hidden			number of hidden neurons
 * @param num_output			number of outputs
 * @return pointer to new fnnfast_data instance
 */
struct fnnfast_data * fnnfast_new(size_t num_input, size_t num_hidden, size_t num_output);

/**
 * Frees the memory space associated with a fnnfast_data instance created by {@link #fnnfast_new(size_t num_input, size_t num_hidden, size_t num_output)}.
 * 
 * @param data      			pointer to fnnfast_data structure
 */
void fnnfast_delete(struct fnnfast_data *data);

/**
 * Calculates and sets the correct pointer addresses used in the fnnfast_data instance.
 * 
 * @param data      			pointer to fnnfast_data structure
 */
void fnnfast_fix_pointers(struct fnnfast_data *data);

/**
 * Zeroes out the neuron, training, and weight data inside the fnnfast_data instance.
 * 
 * @param data      			pointer to fnnfast_data structure
 */
void fnnfast_zero(struct fnnfast_data *data);

/**
 * Psuedorandomly initializes the weights of the fnnfast_data instance.
 * 
 * @param data      			pointer to fnnfast_data structure
 * @param seed      			seed to generate from
 */
void fnnfast_randomize(struct fnnfast_data *data, unsigned int seed);

/**
 * Performs feedforward operation.
 * 
 * @param data      			pointer to fnnfast_data structure
 * @param input     			input to use to estimate output
 * @param output    			output buffer for resulting estimation
 */
void fnnfast_feedforward(struct fnnfast_data *data, neuron *input, neuron *output);

/**
 * Calculates the mean squared deviation or error of the fnnfast_data instance using sets of inputs and expected outputs.
 * 
 * @param data      			pointer to fnnfast_data structure
 * @param input_set 			sets of inputs to test
 * @param output_set			sets of expected outputs
 * @param num       			number of tests
 * @return mean squared 		deviation
 */
neuron fnnfast_mean_squared_deviation(struct fnnfast_data *data, neuron **input_set, neuron **output_set, size_t num);

/**
 * Performs a training round on the fnnfast_data instance.
 * 
 * @param data      			pointer to fnnfast_data structure
 * @param input     			input to train on
 * @param output    			output to train on
 * @param rate      			rate of adjustment
 * @param momentum  			momentum to continue adjustments based on previous adjustments
 * @param training_output_buf	working buffer sizeof(num_output) to use for training
 * @return mean squared deviation
 */
void fnnfast_train(struct fnnfast_data *data, neuron *input, neuron *output, neuron rate, neuron momentum, neuron *training_output_buf);

#endif