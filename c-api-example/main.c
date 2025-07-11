#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("Loading MEF file and running vector addition...\n");
    
    // Initialize MAX runtime
    M_Status *status = M_newStatus();
    M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();
    M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
    if (M_isError(status)) {
        printf("Error creating runtime context: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
        
    // Load the cached model from a MEF file
    M_CompileConfig *compileConfig = M_newCompileConfig();
    M_setModelPath(compileConfig, "graph.mef");

    M_AsyncCompiledModel *compiledModel = M_compileModelSync(context, &compileConfig, status);
    if (M_isError(status)) {
        printf("Error compiling model: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
    
    // Initialize the model
    M_AsyncModel *model = M_initModel(context, compiledModel, NULL, status);
    if (M_isError(status)) {
        printf("Error initializing model: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
    
    // Create input data - two vectors of 8 elements each
    float vector1[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float vector2[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        
    // Create tensor map for inputs
    int64_t shape[1] = {8};
    M_TensorSpec *inputSpec1 = M_newTensorSpec(shape, 1, M_FLOAT32, "input0");
    M_TensorSpec *inputSpec2 = M_newTensorSpec(shape, 1, M_FLOAT32, "input1");

    M_AsyncTensorMap *inputs = M_newAsyncTensorMap(context);
    M_borrowTensorInto(inputs, vector1, inputSpec1, status);
    if (M_isError(status)) {
        printf("Error adding vector1 to tensor map: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
    
    M_borrowTensorInto(inputs, vector2, inputSpec2, status);
    if (M_isError(status)) {
        printf("Error adding vector2 to tensor map: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
    
    // Execute the model
    M_AsyncTensorMap *outputs = M_executeModelSync(context, model, inputs, status);
    if (M_isError(status)) {
        printf("Error executing model: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
    
    // Get the output tensor
    M_AsyncTensor *outputTensor = M_getTensorByNameFrom(outputs, "output0", status);
    if (M_isError(status)) {
        printf("Error getting output tensor: %s\n", M_getError(status));
        return EXIT_FAILURE;
    }
    
    // Get the output data
    const float *outputData = (const float *)M_getTensorData(outputTensor);
    if (outputData == NULL) {
        printf("Error: Output tensor data is NULL\n");
        return EXIT_FAILURE;
    }
        
    // Verify the results
    size_t numElements = M_getTensorNumElements(outputTensor);

    printf("\nInput vectors:\n");
    printf("Vector 1: [");
    for (int i = 0; i < 8; i++) {
        printf("%.1f", vector1[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
    
    printf("Vector 2: [");
    for (int i = 0; i < 8; i++) {
        printf("%.1f", vector2[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
    
    printf("\nOutput vector (%zu elements):\n", numElements);
    printf("Result:   [");
    for (int i = 0; i < 8 && i < numElements; i++) {
        printf("%.1f", outputData[i]);
        if (i < 7) printf(", ");
    }
    printf("]\n");
    
    printf("\nExpected: [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]\n");
    
    int correct = 1;
    for (int i = 0; i < 8; i++) {
        if (outputData[i] != 9.0f) {
            correct = 0;
            break;
        }
    }
    
    if (correct) {
        printf("\nVector addition successful! All results are correct.\n");
    } else {
        printf("\nVector addition failed - results don't match expected values.\n");
    }
    
    // Clean up
    M_freeTensorSpec(inputSpec1);
    M_freeTensorSpec(inputSpec2);
    M_freeAsyncTensorMap(inputs);
    M_freeAsyncTensorMap(outputs);
    M_freeModel(model);
    M_freeCompiledModel(compiledModel);
    M_freeRuntimeConfig(runtimeConfig);
    M_freeRuntimeContext(context);
    M_freeStatus(status);
    
    return EXIT_SUCCESS;
}