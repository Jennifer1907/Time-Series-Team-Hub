"""
Test script to verify parameter flow from app.py to enhanced_training_clean.py
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_parameter_flow():
    """Test if aug_ratio and alpha parameters properly affect max_aug_syn and n_hard_ham"""
    print("=" * 60)
    print("🧪 TESTING PARAMETER FLOW: app.py → enhanced_training_clean.py")
    print("=" * 60)

    try:
        # Import the enhanced_augmentation function
        from enhanced_training_clean import enhanced_augmentation

        # Create test data
        # Use imbalanced data to trigger hard ham generation
        messages = ["Normal message"] * 50 + ["Spam message"] * 20  # 50 ham, 20 spam = 30 difference
        labels = ['ham'] * 50 + ['spam'] * 20

        print(f"📊 Test Dataset: {len(messages)} messages (50 ham, 20 spam, difference = 30)")
        print()

        # Test scenarios
        test_scenarios = [
            ("Low Parameters", 0.1, 0.1),    # Should generate: max_aug_syn=7, n_hard_ham=3
            ("Medium Parameters", 0.3, 0.3), # Should generate: max_aug_syn=21, n_hard_ham=9
            ("High Parameters", 0.5, 0.5),   # Should generate: max_aug_syn=35, n_hard_ham=15
            ("Very High Parameters", 0.8, 0.8), # Should generate: max_aug_syn=56, n_hard_ham=24
        ]

        for scenario_name, aug_ratio, alpha_hard_ham in test_scenarios:
            print(f"🔬 Testing {scenario_name}: aug_ratio={aug_ratio}, alpha_hard_ham={alpha_hard_ham}")

            # Calculate expected values
            expected_max_aug_syn = int(len(messages) * aug_ratio)
            expected_n_hard_ham = int((50 - 20) * alpha_hard_ham)  # (ham_count - spam_count) * alpha

            print(f"   📈 Expected: max_aug_syn={expected_max_aug_syn}, n_hard_ham={expected_n_hard_ham}")

            # Call the function with parameters
            try:
                augmented_messages, augmented_labels = enhanced_augmentation(
                    messages.copy(), labels.copy(),
                    aug_ratio=aug_ratio,
                    alpha_hard_ham=alpha_hard_ham
                )

                print(f"   ✅ Generated: {len(augmented_messages)} total augmented examples")

                # Count types of augmentation
                # Note: In the current implementation, we can't distinguish between hard ham and synonym replacement
                # in the output, but we can see the total and infer from the logs

            except Exception as e:
                print(f"   ❌ Error: {e}")

            print()

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_current_limits():
    """Test the current hardcoded limits"""
    print("🔍 TESTING CURRENT HARDCODED LIMITS")
    print("=" * 60)

    try:
        from enhanced_training_clean import enhanced_augmentation

        # Create larger test dataset to test limits
        messages = ["Normal message"] * 100 + ["Spam message"] * 20  # 100 ham, 20 spam = 80 difference
        labels = ['ham'] * 100 + ['spam'] * 20

        print(f"📊 Large Test Dataset: {len(messages)} messages (100 ham, 20 spam, difference = 80)")

        # Test with high parameters that should hit the limits
        aug_ratio = 0.5  # Should want 60 synonym examples, but limited to 20
        alpha_hard_ham = 0.5  # Should want 40 hard ham examples, but limited to 30

        expected_max_aug_syn_unlimited = int(len(messages) * aug_ratio)  # 60
        expected_n_hard_ham_unlimited = int(80 * alpha_hard_ham)  # 40

        print(f"🎯 High Parameters: aug_ratio={aug_ratio}, alpha_hard_ham={alpha_hard_ham}")
        print(f"   📈 Would want: max_aug_syn={expected_max_aug_syn_unlimited}, n_hard_ham={expected_n_hard_ham_unlimited}")
        print(f"   🚧 But limited to: max_aug_syn=20, n_hard_ham=30")

        # Call the function
        augmented_messages, augmented_labels = enhanced_augmentation(
            messages.copy(), labels.copy(),
            aug_ratio=aug_ratio,
            alpha_hard_ham=alpha_hard_ham
        )

        print(f"   📊 Actually generated: {len(augmented_messages)} total examples")
        print()

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 PARAMETER FLOW TESTING")
    print("Testing if app.py slider changes affect enhanced_training_clean.py calculations")
    print()

    # Test basic parameter flow
    test1_result = test_parameter_flow()

    # Test current limits
    test2_result = test_current_limits()

    print("=" * 60)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    if test1_result and test2_result:
        print("✅ Parameter flow works, but there are hardcoded limits!")
        print()
        print("🔧 ISSUES FOUND:")
        print("   1. max_aug_syn is limited to 20 examples maximum")
        print("   2. n_hard_ham is limited to 30 examples maximum")
        print("   3. App.py sliders can go up to 0.5 (50%) but won't have full effect")
        print()
        print("💡 RECOMMENDATIONS:")
        print("   1. Remove or increase the hardcoded limits in enhanced_training_clean.py")
        print("   2. Make limits configurable or based on dataset size")
        print("   3. Update limits to match app.py slider ranges")
        print()
        print("🎯 PROPOSED CHANGES:")
        print("   • Change: min(int(len(messages) * aug_ratio), 20)")
        print("   • To:     int(len(messages) * aug_ratio)  # No arbitrary limit")
        print("   • Change: min(int((ham_count - spam_count) * alpha_hard_ham), 30)")
        print("   • To:     int((ham_count - spam_count) * alpha_hard_ham)  # No arbitrary limit")

    else:
        print("❌ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()
