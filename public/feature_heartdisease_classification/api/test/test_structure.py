"""
Test import structure without loading heavy dependencies
"""
import sys
import os

print("="*70)
print("IMPORT STRUCTURE TEST")
print("="*70)
print()

# Test 1: Check if utility is a package
print("1. Checking if utility/ is a proper package...")
utility_init = os.path.join('utility', '__init__.py')
if os.path.exists(utility_init):
    print(f"   ✓ {utility_init} exists")
else:
    print(f"   ✗ {utility_init} NOT FOUND")
    print("   Create it with: touch utility/__init__.py")
    sys.exit(1)

# Test 2: Check all required files exist
print("\n2. Checking if all utility files exist...")
required_files = [
    'utility/model.py',
    'utility/load_model.py',
    'utility/load_data.py',
    'utility/feature_engineer.py',
    'utility/preprocessing.py',
    'utility/predict.py'
]

all_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} NOT FOUND")
        all_exist = False

if not all_exist:
    print("\n   Some files are missing!")
    sys.exit(1)

# Test 3: Check for correct import syntax
print("\n3. Checking import syntax in utility files...")

import_checks = {
    'utility/load_model.py': 'from .model import',
    'utility/predict.py': 'from .preprocessing import',
    'utility/preprocessing.py': 'from .feature_engineer import'
}

for file, expected_import in import_checks.items():
    with open(file, 'r') as f:
        content = f.read()
        if expected_import in content:
            print(f"   ✓ {file} uses relative imports")
        else:
            print(f"   ✗ {file} may have wrong imports")
            print(f"      Expected to find: {expected_import}")

# Test 4: Try to import the package
print("\n4. Attempting to import utility package...")
try:
    import utility
    print(f"   ✓ Successfully imported utility package")
    print(f"   ✓ Package location: {utility.__file__}")
except ImportError as e:
    print(f"   ✗ Failed to import utility: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("IMPORT STRUCTURE TEST PASSED!")
print("="*70)
print()
print("Next steps:")
print("1. Install dependencies: pip install -r requirements.txt")
print("2. Ensure model files exist in ../models/")
print("3. Run: python main.py")
print()
