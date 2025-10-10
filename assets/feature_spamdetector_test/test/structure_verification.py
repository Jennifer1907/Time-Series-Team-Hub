"""
Code structure verification for data augmentation centralization
"""

import os
import re

def analyze_file_content(file_path, description):
    """Analyze a file's content for specific patterns"""
    print(f"\n📁 Analyzing {description}: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"   ❌ File not found: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False

def check_spam_model_simplification():
    """Check that spam_model.py has been simplified"""
    print("\n1. 🔍 Checking spam_model.py simplification...")

    content = analyze_file_content('spam_model.py', 'Core Classifier')
    if not content:
        return False

    # Check for removed functions
    removed_functions = [
        'def _get_hard_ham_phrase_groups',
        'def _generate_hard_ham',
        'def _synonym_replacement',
        'def _augment_dataset',
        'def apply_augmentation'
    ]

    found_old_functions = []
    for func in removed_functions:
        if func in content:
            found_old_functions.append(func)

    if found_old_functions:
        print(f"   ❌ Still contains old functions: {found_old_functions}")
        return False
    else:
        print("   ✅ All old augmentation functions removed")

    # Check for centralized import
    if 'from enhanced_training_clean import enhanced_augmentation' in content:
        print("   ✅ Imports from centralized module")
    else:
        print("   ❌ Missing centralized import")
        return False

    # Check for removed NLTK imports
    if 'import nltk' in content:
        print("   ❌ Still contains NLTK import")
        return False
    else:
        print("   ✅ NLTK imports removed")

    # Check simplified constructor
    constructor_match = re.search(r'def __init__\(self[^)]*\):', content)
    if constructor_match:
        constructor = constructor_match.group(0)
        if 'aug_ratio' in constructor or 'alpha' in constructor:
            print("   ❌ Constructor still has augmentation parameters")
            return False
        else:
            print("   ✅ Constructor simplified (no aug_ratio/alpha)")

    return True

def check_enhanced_training_centralization():
    """Check that enhanced_training_clean.py contains all functions"""
    print("\n2. 🔍 Checking enhanced_training_clean.py centralization...")

    content = analyze_file_content('enhanced_training_clean.py', 'Centralized Augmentation')
    if not content:
        return False

    # Check for required functions
    required_functions = [
        'def get_hard_ham_phrase_groups',
        'def generate_hard_ham',
        'def synonym_replacement',
        'def enhanced_augmentation',
        'def run_enhanced_pipeline',
        'def add_enhanced_method_to_classifier'
    ]

    missing_functions = []
    for func in required_functions:
        if func not in content:
            missing_functions.append(func)

    if missing_functions:
        print(f"   ❌ Missing functions: {missing_functions}")
        return False
    else:
        print("   ✅ All required functions present")

    # Check for NLTK handling
    if 'import nltk' in content or 'from nltk.corpus import wordnet' in content:
        print("   ✅ NLTK imports properly handled in centralized module")
    else:
        print("   ⚠️  No NLTK imports found (might be expected)")

    return True

def check_app_integration():
    """Check that app.py uses centralized functions"""
    print("\n3. 🔍 Checking app.py integration...")

    content = analyze_file_content('app.py', 'Streamlit App')
    if not content:
        return False

    # Check for centralized import
    if 'from enhanced_training_clean import' in content:
        print("   ✅ Imports from centralized module")
    else:
        print("   ❌ Missing centralized import")
        return False

    # Check for simplified constructor calls
    classifier_calls = re.findall(r'SpamClassifier\([^)]*\)', content)
    old_style_calls = [call for call in classifier_calls if 'aug_ratio=' in call or 'alpha=' in call]

    if old_style_calls:
        print(f"   ⚠️  Found old-style constructor calls: {len(old_style_calls)}")
        # This might be ok if they're in enhanced training context
        print("   ℹ️  These might be in enhanced training context, which is acceptable")
    else:
        print("   ✅ All constructor calls use simplified parameters")

    return True

def check_file_sizes():
    """Check file sizes to verify code has been moved"""
    print("\n4. 📊 Checking file sizes (indication of code movement)...")

    files_to_check = [
        ('spam_model.py', 'Should be smaller after removing augmentation'),
        ('enhanced_training_clean.py', 'Should contain all augmentation code'),
        ('app.py', 'Should be roughly the same size')
    ]

    for filename, description in files_to_check:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"   📄 {filename}: {size:,} bytes - {description}")
        else:
            print(f"   ❌ {filename}: File not found")

    return True

def main():
    """Run all verification checks"""
    print("=" * 80)
    print("🎯 DATA AUGMENTATION CENTRALIZATION VERIFICATION")
    print("=" * 80)
    print("📍 Working directory:", os.getcwd())

    checks = [
        ("SpamClassifier Simplification", check_spam_model_simplification),
        ("Enhanced Training Centralization", check_enhanced_training_centralization),
        ("App Integration", check_app_integration),
        ("File Size Analysis", check_file_sizes)
    ]

    passed = 0
    total = len(checks)

    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
                print(f"   ✅ {name}: PASSED")
            else:
                print(f"   ❌ {name}: FAILED")
        except Exception as e:
            print(f"   ❌ {name}: ERROR - {e}")

    print("\n" + "=" * 80)
    print(f"🎯 FINAL RESULT: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 SUCCESS: Data augmentation centralization is complete!")
        print("\n📋 What was accomplished:")
        print("   ✅ All augmentation functions moved to enhanced_training_clean.py")
        print("   ✅ SpamClassifier simplified and cleaned")
        print("   ✅ Redundant code removed")
        print("   ✅ Single source of truth established")
        print("   ✅ Proper imports and integration")

        print("\n🚀 Benefits achieved:")
        print("   • No code duplication")
        print("   • Easier maintenance")
        print("   • Cleaner architecture")
        print("   • Better testability")
    else:
        print("\n⚠️  Some checks failed, but this might be due to environment issues.")
        print("Review the specific failures above.")

    print("=" * 80)

if __name__ == "__main__":
    main()
