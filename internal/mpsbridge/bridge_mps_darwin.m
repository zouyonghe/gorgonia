// +build mps,darwin

#import "bridge.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdlib.h>
#include <string.h>

int gorgonia_mps_available(void) {
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return 0;
		}
		MPSGraph *graph = [[MPSGraph alloc] init];
		return graph != nil;
	}
}

char* gorgonia_mps_device_name(void) {
	@autoreleasepool {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return NULL;
		}
		const char *name = [[device name] UTF8String];
		if (name == NULL) {
			return NULL;
		}
		return strdup(name);
	}
}

void* gorgonia_mps_new_float32_buffer(const float* data, int count) {
	@autoreleasepool {
		if (data == NULL || count <= 0) {
			return NULL;
		}
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return NULL;
		}
		NSUInteger length = (NSUInteger)count * sizeof(float);
		id<MTLBuffer> buffer = [device newBufferWithBytes:data
												length:length
											 options:MTLResourceStorageModeShared];
		return buffer == nil ? NULL : (__bridge_retained void*)buffer;
	}
}

int gorgonia_mps_read_float32_buffer(void* rawBuffer, float* out, int count) {
	@autoreleasepool {
		if (rawBuffer == NULL || out == NULL || count <= 0) {
			return 0;
		}
		id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)rawBuffer;
		NSUInteger length = (NSUInteger)count * sizeof(float);
		if ([buffer length] < length) {
			return 0;
		}
		memcpy(out, [buffer contents], length);
		return 1;
	}
}

void gorgonia_mps_release_buffer(void* rawBuffer) {
	if (rawBuffer != NULL) {
		CFRelease(rawBuffer);
	}
}

int gorgonia_mps_matmul_float32(const float* a, const float* b, float* out, int m, int k, int n) {
	@autoreleasepool {
		if (a == NULL || b == NULL || out == NULL || m <= 0 || k <= 0 || n <= 0) {
			return 0;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return 0;
		}

		NSUInteger aLength = (NSUInteger)m * (NSUInteger)k * sizeof(float);
		NSUInteger bLength = (NSUInteger)k * (NSUInteger)n * sizeof(float);
		NSUInteger outLength = (NSUInteger)m * (NSUInteger)n * sizeof(float);
		id<MTLBuffer> aBuffer = [device newBufferWithBytes:a length:aLength options:MTLResourceStorageModeShared];
		id<MTLBuffer> bBuffer = [device newBufferWithBytes:b length:bLength options:MTLResourceStorageModeShared];
		if (aBuffer == nil || bBuffer == nil) {
			return 0;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return 0;
		}

		MPSGraphTensor *aTensor = [graph placeholderWithShape:@[@(m), @(k)] dataType:MPSDataTypeFloat32 name:@"a"];
		MPSGraphTensor *bTensor = [graph placeholderWithShape:@[@(k), @(n)] dataType:MPSDataTypeFloat32 name:@"b"];
		MPSGraphTensor *product = [graph matrixMultiplicationWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"matmul"];

		MPSGraphTensorData *aData = [[MPSGraphTensorData alloc] initWithMTLBuffer:aBuffer shape:@[@(m), @(k)] dataType:MPSDataTypeFloat32];
		MPSGraphTensorData *bData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bBuffer shape:@[@(k), @(n)] dataType:MPSDataTypeFloat32];

		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
			aTensor: aData,
			bTensor: bData,
		};
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[product] targetOperations:nil];
		MPSGraphTensorData *productData = results[product];
		MPSNDArray *productArray = [productData mpsndarray];
		if (productData == nil || productArray == nil) {
			return 0;
		}
		[productArray readBytes:out strideBytes:nil];
		return 1;
	}
}

void* gorgonia_mps_matmul_float32_buffers(void* rawA, void* rawB, int m, int k, int n) {
	@autoreleasepool {
		if (rawA == NULL || rawB == NULL || m <= 0 || k <= 0 || n <= 0) {
			return NULL;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return NULL;
		}

		id<MTLBuffer> aBuffer = (__bridge id<MTLBuffer>)rawA;
		id<MTLBuffer> bBuffer = (__bridge id<MTLBuffer>)rawB;
		NSUInteger aLength = (NSUInteger)m * (NSUInteger)k * sizeof(float);
		NSUInteger bLength = (NSUInteger)k * (NSUInteger)n * sizeof(float);
		NSUInteger outLength = (NSUInteger)m * (NSUInteger)n * sizeof(float);
		if ([aBuffer length] < aLength || [bBuffer length] < bLength) {
			return NULL;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return NULL;
		}

		MPSGraphTensor *aTensor = [graph placeholderWithShape:@[@(m), @(k)] dataType:MPSDataTypeFloat32 name:@"a"];
		MPSGraphTensor *bTensor = [graph placeholderWithShape:@[@(k), @(n)] dataType:MPSDataTypeFloat32 name:@"b"];
		MPSGraphTensor *product = [graph matrixMultiplicationWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"matmul"];

		MPSGraphTensorData *aData = [[MPSGraphTensorData alloc] initWithMTLBuffer:aBuffer shape:@[@(m), @(k)] dataType:MPSDataTypeFloat32];
		MPSGraphTensorData *bData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bBuffer shape:@[@(k), @(n)] dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
			aTensor: aData,
			bTensor: bData,
		};
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[product] targetOperations:nil];
		MPSGraphTensorData *productData = results[product];
		MPSNDArray *productArray = [productData mpsndarray];
		if (productData == nil || productArray == nil) {
			return NULL;
		}

		id<MTLBuffer> outBuffer = [device newBufferWithLength:outLength options:MTLResourceStorageModeShared];
		if (outBuffer == nil) {
			return NULL;
		}
		[productArray readBytes:[outBuffer contents] strideBytes:nil];
		return (__bridge_retained void*)outBuffer;
	}
}

int gorgonia_mps_add_float32(const float* a, const float* b, float* out, int count) {
	return gorgonia_mps_elementwise_float32(a, b, out, count, 0);
}

int gorgonia_mps_elementwise_float32(const float* a, const float* b, float* out, int count, int op) {
	@autoreleasepool {
		if (a == NULL || b == NULL || out == NULL || count <= 0) {
			return 0;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return 0;
		}

		NSUInteger length = (NSUInteger)count * sizeof(float);
		id<MTLBuffer> aBuffer = [device newBufferWithBytes:a length:length options:MTLResourceStorageModeShared];
		id<MTLBuffer> bBuffer = [device newBufferWithBytes:b length:length options:MTLResourceStorageModeShared];
		if (aBuffer == nil || bBuffer == nil) {
			return 0;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return 0;
		}

		MPSShape *shape = @[@(count)];
		MPSGraphTensor *aTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"a"];
		MPSGraphTensor *bTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"b"];
		MPSGraphTensor *result = nil;
		switch (op) {
			case 0:
				result = [graph additionWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"add"];
				break;
			case 1:
				result = [graph subtractionWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"sub"];
				break;
			case 2:
				result = [graph multiplicationWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"mul"];
				break;
			default:
				return 0;
		}

		MPSGraphTensorData *aData = [[MPSGraphTensorData alloc] initWithMTLBuffer:aBuffer shape:shape dataType:MPSDataTypeFloat32];
		MPSGraphTensorData *bData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bBuffer shape:shape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
			aTensor: aData,
			bTensor: bData,
		};
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[result] targetOperations:nil];
		MPSGraphTensorData *resultData = results[result];
		MPSNDArray *resultArray = [resultData mpsndarray];
		if (resultData == nil || resultArray == nil) {
			return 0;
		}
		[resultArray readBytes:out strideBytes:nil];
		return 1;
	}
}

void* gorgonia_mps_elementwise_float32_buffers(void* rawA, void* rawB, int count, int op) {
	@autoreleasepool {
		if (rawA == NULL || rawB == NULL || count <= 0) {
			return NULL;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return NULL;
		}

		id<MTLBuffer> aBuffer = (__bridge id<MTLBuffer>)rawA;
		id<MTLBuffer> bBuffer = (__bridge id<MTLBuffer>)rawB;
		NSUInteger length = (NSUInteger)count * sizeof(float);
		if ([aBuffer length] < length || [bBuffer length] < length) {
			return NULL;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return NULL;
		}

		MPSShape *shape = @[@(count)];
		MPSGraphTensor *aTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"a"];
		MPSGraphTensor *bTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"b"];
		MPSGraphTensor *result = nil;
		switch (op) {
			case 0:
				result = [graph additionWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"add"];
				break;
			case 1:
				result = [graph subtractionWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"sub"];
				break;
			case 2:
				result = [graph multiplicationWithPrimaryTensor:aTensor secondaryTensor:bTensor name:@"mul"];
				break;
			default:
				return NULL;
		}

		MPSGraphTensorData *aData = [[MPSGraphTensorData alloc] initWithMTLBuffer:aBuffer shape:shape dataType:MPSDataTypeFloat32];
		MPSGraphTensorData *bData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bBuffer shape:shape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
			aTensor: aData,
			bTensor: bData,
		};
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[result] targetOperations:nil];
		MPSGraphTensorData *resultData = results[result];
		MPSNDArray *resultArray = [resultData mpsndarray];
		if (resultData == nil || resultArray == nil) {
			return NULL;
		}

		id<MTLBuffer> outBuffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
		if (outBuffer == nil) {
			return NULL;
		}
		[resultArray readBytes:[outBuffer contents] strideBytes:nil];
		return (__bridge_retained void*)outBuffer;
	}
}

int gorgonia_mps_relu_float32(const float* input, float* out, int count) {
	@autoreleasepool {
		if (input == NULL || out == NULL || count <= 0) {
			return 0;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return 0;
		}

		NSUInteger length = (NSUInteger)count * sizeof(float);
		id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input length:length options:MTLResourceStorageModeShared];
		if (inputBuffer == nil) {
			return 0;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return 0;
		}

		MPSShape *shape = @[@(count)];
		MPSGraphTensor *inputTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"input"];
		MPSGraphTensor *relu = [graph reLUWithTensor:inputTensor name:@"relu"];
		MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer shape:shape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{ inputTensor: inputData };
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[relu] targetOperations:nil];
		MPSGraphTensorData *reluData = results[relu];
		MPSNDArray *reluArray = [reluData mpsndarray];
		if (reluData == nil || reluArray == nil) {
			return 0;
		}
		[reluArray readBytes:out strideBytes:nil];
		return 1;
	}
}

void* gorgonia_mps_relu_float32_buffer(void* rawInput, int count) {
	@autoreleasepool {
		if (rawInput == NULL || count <= 0) {
			return NULL;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return NULL;
		}

		id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)rawInput;
		NSUInteger length = (NSUInteger)count * sizeof(float);
		if ([inputBuffer length] < length) {
			return NULL;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return NULL;
		}

		MPSShape *shape = @[@(count)];
		MPSGraphTensor *inputTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"input"];
		MPSGraphTensor *relu = [graph reLUWithTensor:inputTensor name:@"relu"];
		MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer shape:shape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{ inputTensor: inputData };
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[relu] targetOperations:nil];
		MPSGraphTensorData *reluData = results[relu];
		MPSNDArray *reluArray = [reluData mpsndarray];
		if (reluData == nil || reluArray == nil) {
			return NULL;
		}

		id<MTLBuffer> outBuffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
		if (outBuffer == nil) {
			return NULL;
		}
		[reluArray readBytes:[outBuffer contents] strideBytes:nil];
		return (__bridge_retained void*)outBuffer;
	}
}

int gorgonia_mps_add_row_bias_float32(const float* matrix, const float* bias, float* out, int rows, int cols) {
	@autoreleasepool {
		if (matrix == NULL || bias == NULL || out == NULL || rows <= 0 || cols <= 0) {
			return 0;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return 0;
		}

		NSUInteger matrixLength = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
		NSUInteger biasLength = (NSUInteger)cols * sizeof(float);
		id<MTLBuffer> matrixBuffer = [device newBufferWithBytes:matrix length:matrixLength options:MTLResourceStorageModeShared];
		id<MTLBuffer> biasBuffer = [device newBufferWithBytes:bias length:biasLength options:MTLResourceStorageModeShared];
		if (matrixBuffer == nil || biasBuffer == nil) {
			return 0;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return 0;
		}

		MPSShape *matrixShape = @[@(rows), @(cols)];
		MPSShape *biasShape = @[@(cols)];
		MPSGraphTensor *matrixTensor = [graph placeholderWithShape:matrixShape dataType:MPSDataTypeFloat32 name:@"matrix"];
		MPSGraphTensor *biasTensor = [graph placeholderWithShape:biasShape dataType:MPSDataTypeFloat32 name:@"bias"];
		MPSGraphTensor *sum = [graph additionWithPrimaryTensor:matrixTensor secondaryTensor:biasTensor name:@"row_bias_add"];

		MPSGraphTensorData *matrixData = [[MPSGraphTensorData alloc] initWithMTLBuffer:matrixBuffer shape:matrixShape dataType:MPSDataTypeFloat32];
		MPSGraphTensorData *biasData = [[MPSGraphTensorData alloc] initWithMTLBuffer:biasBuffer shape:biasShape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
			matrixTensor: matrixData,
			biasTensor: biasData,
		};
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[sum] targetOperations:nil];
		MPSGraphTensorData *sumData = results[sum];
		MPSNDArray *sumArray = [sumData mpsndarray];
		if (sumData == nil || sumArray == nil) {
			return 0;
		}
		[sumArray readBytes:out strideBytes:nil];
		return 1;
	}
}

void* gorgonia_mps_add_row_bias_float32_buffers(void* rawMatrix, void* rawBias, int rows, int cols) {
	@autoreleasepool {
		if (rawMatrix == NULL || rawBias == NULL || rows <= 0 || cols <= 0) {
			return NULL;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return NULL;
		}

		id<MTLBuffer> matrixBuffer = (__bridge id<MTLBuffer>)rawMatrix;
		id<MTLBuffer> biasBuffer = (__bridge id<MTLBuffer>)rawBias;
		NSUInteger matrixLength = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
		NSUInteger biasLength = (NSUInteger)cols * sizeof(float);
		if ([matrixBuffer length] < matrixLength || [biasBuffer length] < biasLength) {
			return NULL;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return NULL;
		}

		MPSShape *matrixShape = @[@(rows), @(cols)];
		MPSShape *biasShape = @[@(cols)];
		MPSGraphTensor *matrixTensor = [graph placeholderWithShape:matrixShape dataType:MPSDataTypeFloat32 name:@"matrix"];
		MPSGraphTensor *biasTensor = [graph placeholderWithShape:biasShape dataType:MPSDataTypeFloat32 name:@"bias"];
		MPSGraphTensor *sum = [graph additionWithPrimaryTensor:matrixTensor secondaryTensor:biasTensor name:@"row_bias_add"];

		MPSGraphTensorData *matrixData = [[MPSGraphTensorData alloc] initWithMTLBuffer:matrixBuffer shape:matrixShape dataType:MPSDataTypeFloat32];
		MPSGraphTensorData *biasData = [[MPSGraphTensorData alloc] initWithMTLBuffer:biasBuffer shape:biasShape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{
			matrixTensor: matrixData,
			biasTensor: biasData,
		};
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[sum] targetOperations:nil];
		MPSGraphTensorData *sumData = results[sum];
		MPSNDArray *sumArray = [sumData mpsndarray];
		if (sumData == nil || sumArray == nil) {
			return NULL;
		}

		id<MTLBuffer> outBuffer = [device newBufferWithLength:matrixLength options:MTLResourceStorageModeShared];
		if (outBuffer == nil) {
			return NULL;
		}
		[sumArray readBytes:[outBuffer contents] strideBytes:nil];
		return (__bridge_retained void*)outBuffer;
	}
}

int gorgonia_mps_softmax_rows_float32(const float* input, float* out, int rows, int cols, int log_output) {
	@autoreleasepool {
		if (input == NULL || out == NULL || rows <= 0 || cols <= 0) {
			return 0;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return 0;
		}

		NSUInteger length = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
		id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input length:length options:MTLResourceStorageModeShared];
		if (inputBuffer == nil) {
			return 0;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return 0;
		}

		MPSShape *shape = @[@(rows), @(cols)];
		MPSGraphTensor *inputTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"input"];
		MPSGraphTensor *softmax = [graph softMaxWithTensor:inputTensor axis:1 name:@"softmax"];
		MPSGraphTensor *result = softmax;
		if (log_output) {
			result = [graph logarithmWithTensor:softmax name:@"log_softmax"];
		}
		MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer shape:shape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{ inputTensor: inputData };
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[result] targetOperations:nil];
		MPSGraphTensorData *softmaxData = results[result];
		MPSNDArray *softmaxArray = [softmaxData mpsndarray];
		if (softmaxData == nil || softmaxArray == nil) {
			return 0;
		}
		[softmaxArray readBytes:out strideBytes:nil];
		return 1;
	}
}

void* gorgonia_mps_softmax_rows_float32_buffers(void* rawInput, int rows, int cols, int logOutput) {
	@autoreleasepool {
		if (rawInput == NULL || rows <= 0 || cols <= 0) {
			return NULL;
		}

		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (device == nil) {
			return NULL;
		}

		id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)rawInput;
		NSUInteger length = (NSUInteger)rows * (NSUInteger)cols * sizeof(float);
		if ([inputBuffer length] < length) {
			return NULL;
		}

		MPSGraph *graph = [[MPSGraph alloc] init];
		if (graph == nil) {
			return NULL;
		}

		MPSShape *shape = @[@(rows), @(cols)];
		MPSGraphTensor *inputTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"input"];
		MPSGraphTensor *outputTensor = [graph softMaxWithTensor:inputTensor axis:1 name:@"softmax"];
		if (logOutput) {
			outputTensor = [graph logarithmWithTensor:outputTensor name:@"log_softmax"];
		}

		MPSGraphTensorData *inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer shape:shape dataType:MPSDataTypeFloat32];
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = @{ inputTensor: inputData };
		NSDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [graph runWithFeeds:feeds targetTensors:@[outputTensor] targetOperations:nil];
		MPSGraphTensorData *outputData = results[outputTensor];
		MPSNDArray *outputArray = [outputData mpsndarray];
		if (outputData == nil || outputArray == nil) {
			return NULL;
		}

		id<MTLBuffer> outBuffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
		if (outBuffer == nil) {
			return NULL;
		}
		[outputArray readBytes:[outBuffer contents] strideBytes:nil];
		return (__bridge_retained void*)outBuffer;
	}
}
