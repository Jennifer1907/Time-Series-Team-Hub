"""
Quick test to verify all imports work correctly
"""
print("Testing utility package imports...")

try:
    print("1. Testing model imports...")
    from utility.model import NumericalMLP, FeatureFusionModel
    print("   ✓ Model imports successful")

    print("2. Testing load_model imports...")
    from utility.load_model import fusion_model, num_scaler, cnn_scaler
    print("   ✓ Load_model imports successful")
    print(f"   - Fusion model loaded: {fusion_model is not None}")
    print(f"   - Num scaler loaded: {num_scaler is not None}")
    print(f"   - CNN scaler loaded: {cnn_scaler is not None}")

    print("3. Testing predict imports...")
    from utility.predict import predict_single_sample
    print("   ✓ Predict imports successful")

    print("\n" + "="*50)
    print("ALL IMPORTS SUCCESSFUL!")
    print("="*50)
    print("\nYou can now run: python main.py")

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're running from the 'api' directory")
    print("2. Check that all files exist in the utility folder")
    print("3. Verify model files exist in ../models/")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
