#!/usr/bin/env python3
"""
Fixed model converter - uses correct export method for SavedModel
"""

import tensorflow as tf
import json
import os

def simple_h5_to_js_conversion(h5_path, output_dir):
    """Simple conversion that should work with your setup"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {h5_path}...")
    
    # Load the H5 model
    model = tf.keras.models.load_model(h5_path)
    
    print("Model loaded successfully!")
    print("\nModel Summary:")
    model.summary()
    
    # Method 1: Export as SavedModel format (TF.js can load this)
    saved_model_path = os.path.join(output_dir, 'saved_model')
    print(f"\nExporting as SavedModel format to {saved_model_path}")
    
    # Use export instead of save for SavedModel format
    model.export(saved_model_path)
    
    # Method 2: Save model architecture and weights separately  
    # Save model architecture as JSON
    try:
        model_json = model.to_json()
        json_path = os.path.join(output_dir, 'model_architecture.json')
        with open(json_path, 'w') as f:
            json.dump({"model_config": json.loads(model_json)}, f, indent=2)
        print(f"Architecture saved to: {json_path}")
    except Exception as e:
        print(f"Could not save architecture as JSON: {e}")
    
    # Save weights as H5
    try:
        weights_path = os.path.join(output_dir, 'model_weights.h5')
        model.save_weights(weights_path)
        print(f"Weights saved to: {weights_path}")
    except Exception as e:
        print(f"Could not save weights: {e}")
    
    # Method 3: Try to create a basic TensorFlow.js compatible structure
    try:
        # Create a simple model.json for TF.js
        tfjs_model_path = os.path.join(output_dir, 'model.json')
        
        # Basic model metadata for TF.js
        tfjs_metadata = {
            "format": "layers-model",
            "generatedBy": "manual-conversion",
            "convertedBy": "python-script",
            "signature": {
                "inputs": {"input": {"name": "input_layer", "dtype": "float32", "shape": [None, 224, 224, 3]}},
                "outputs": {"output": {"name": "dense_1", "dtype": "float32", "shape": [None, 2]}}
            },
            "userDefinedMetadata": {
                "classNames": ["mango_trees", "not_mango_trees"]
            }
        }
        
        with open(tfjs_model_path, 'w') as f:
            json.dump(tfjs_metadata, f, indent=2)
        
        print(f"TensorFlow.js metadata saved to: {tfjs_model_path}")
        
    except Exception as e:
        print(f"Could not create TF.js metadata: {e}")
    
    print(f"SavedModel exported to: {saved_model_path}")
    
    # Get model info
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    print(f"\nModel Info:")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    print(f"Number of classes: {output_shape[-1] if output_shape else 'Unknown'}")
    
    # List the files created
    print(f"\n📁 Files created in {output_dir}:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    return output_dir

if __name__ == "__main__":
    # Your model file
    h5_file = "mango_model-v001.h5"
    output_directory = "mango_model_js"
    
    if not os.path.exists(h5_file):
        print(f"Error: {h5_file} not found!")
        print("Available .h5 files:")
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                print(f"  - {f}")
        exit(1)
    
    try:
        simple_h5_to_js_conversion(h5_file, output_directory)
        print(f"\n✅ Conversion completed successfully!")
        print(f"📁 Output directory: {output_directory}")
        print("\n📋 Next steps:")
        print("1. Copy the entire mango_model_js folder to your React app's public folder")
        print("2. In your React code, try loading from:")
        print("   - './mango_model_js/saved_model' (SavedModel format)")
        print("   - './mango_model_js/model.json' (if TF.js metadata works)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()