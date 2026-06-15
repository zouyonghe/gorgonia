#ifndef GORGONIA_MPS_BRIDGE_H
#define GORGONIA_MPS_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

int gorgonia_mps_available(void);
char* gorgonia_mps_device_name(void);
void* gorgonia_mps_new_float32_buffer(const float* data, int count);
int gorgonia_mps_read_float32_buffer(void* buffer, float* out, int count);
void gorgonia_mps_release_buffer(void* buffer);
int gorgonia_mps_matmul_float32(const float* a, const float* b, float* out, int m, int k, int n);
int gorgonia_mps_add_float32(const float* a, const float* b, float* out, int count);
int gorgonia_mps_elementwise_float32(const float* a, const float* b, float* out, int count, int op);
int gorgonia_mps_relu_float32(const float* input, float* out, int count);
int gorgonia_mps_add_row_bias_float32(const float* matrix, const float* bias, float* out, int rows, int cols);
int gorgonia_mps_softmax_rows_float32(const float* input, float* out, int rows, int cols, int log_output);

#ifdef __cplusplus
}
#endif

#endif
