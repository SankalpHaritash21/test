import tensorflow as tf

# Check number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# List all physical devices (CPU and GPU)
print("\nPhysical Devices:")
print(tf.config.list_physical_devices())

# Check detailed GPU information
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\nGPU Details:")
    for gpu in gpus:
        print(f"Device Name: {gpu.name}")
        print(f"Device Type: {gpu.device_type}")

# Check if TensorFlow is built with GPU support
if tf.test.is_built_with_cuda():
    print("\nTensorFlow is built with GPU (CUDA) support.")
else:
    print("\nTensorFlow is NOT built with GPU (CUDA) support.")

# Test if TensorFlow is using the GPU
if tf.test.gpu_device_name():
    print(f"\nDefault GPU Device: {tf.test.gpu_device_name()}")
else:
    print("\nNo GPU detected or TensorFlow is not using the GPU.")
