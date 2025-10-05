#!/usr/bin/env python3
"""
🎯 TensorFlow Configuration Fix

Fixes the TensorFlow eager execution warning by using the correct TensorFlow 2.x APIs.
"""

import tensorflow as tf
import os

def fix_tensorflow_configuration():
    """Fix TensorFlow configuration to eliminate warnings"""
    
    # Set memory growth for GPU (if available)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass  # GPU not available or already configured
    
    # Enable eager execution (TensorFlow 2.x default, but ensure it's on)
    tf.config.run_functions_eagerly(True)
    
    # Enable debug mode for tf.data functions (fixes the warning)
    tf.data.experimental.enable_debug_mode()
    
    # Suppress oneDNN optimization warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # Suppress other TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
    
    print("✅ TensorFlow configuration optimized")

def configure_tensorflow_for_production():
    """Production-optimized TensorFlow configuration"""
    
    # Set memory growth and limits
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Optional: Set memory limit (uncomment and adjust as needed)
                # tf.config.experimental.set_memory_limit(gpu, 1024)  # MB
    except Exception:
        pass
    
    # Optimize for inference
    tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
    
    # Enable mixed precision (if supported)
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled for faster inference")
    except Exception:
        print("ℹ️ Mixed precision not available on this system")
    
    # Configure threading for CPU
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
    
    print("✅ TensorFlow production configuration applied")

if __name__ == "__main__":
    print("🔧 Configuring TensorFlow...")
    fix_tensorflow_configuration()
    configure_tensorflow_for_production()
    print("🎉 TensorFlow configuration complete!")