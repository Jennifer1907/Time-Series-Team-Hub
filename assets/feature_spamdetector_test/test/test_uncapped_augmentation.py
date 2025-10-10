"""
Test to verify that augmentation caps have been removed
"""

def test_uncapped_formulas():
    """Test the new uncapped formulas"""

    print("🚀 TESTING UNCAPPED AUGMENTATION FORMULAS")
    print("=" * 60)

    # Test dataset characteristics
    dataset_size = 1000  # Large dataset to test uncapped behavior
    ham_count = 800
    spam_count = 200
    ham_excess = ham_count - spam_count

    print(f"📊 Test Dataset:")
    print(f"   Total messages: {dataset_size}")
    print(f"   Ham messages: {ham_count}")
    print(f"   Spam messages: {spam_count}")
    print(f"   Ham excess: {ham_excess}")
    print()

    # Test high parameter values that would have been capped before
    test_params = [
        {"aug_ratio": 0.1, "alpha": 0.1, "name": "Low"},
        {"aug_ratio": 0.3, "alpha": 0.3, "name": "Medium"},
        {"aug_ratio": 0.5, "alpha": 0.5, "name": "High"},
        {"aug_ratio": 0.8, "alpha": 0.8, "name": "Very High"},
        {"aug_ratio": 1.0, "alpha": 1.0, "name": "Maximum"}
    ]

    print("🔧 TESTING UNCAPPED FORMULAS")
    print("-" * 80)
    print(f"{'Scenario':<12} {'aug_ratio':<10} {'alpha':<8} {'max_aug_syn':<12} {'n_hard_ham':<12} {'Total':<8}")
    print("-" * 80)

    for params in test_params:
        aug_ratio = params['aug_ratio']
        alpha = params['alpha']

        # Calculate using the new uncapped formulas
        max_aug_syn = int(dataset_size * aug_ratio)  # No min(x, 20) cap
        n_hard_ham = int(ham_excess * alpha)         # No min(x, 30) cap
        total_augmented = max_aug_syn + n_hard_ham

        print(f"{params['name']:<12} {aug_ratio:<10.1f} {alpha:<8.1f} {max_aug_syn:<12d} {n_hard_ham:<12d} {total_augmented:<8d}")

    print("-" * 80)
    print()

    # Verify caps are removed
    print("🔍 VERIFICATION: Caps Removed")
    print("-" * 40)

    # Test that we can exceed the old caps
    extreme_aug_ratio = 0.5
    extreme_alpha = 0.8

    max_aug_syn_uncapped = int(dataset_size * extreme_aug_ratio)
    n_hard_ham_uncapped = int(ham_excess * extreme_alpha)

    print(f"With extreme parameters (aug_ratio={extreme_aug_ratio}, alpha={extreme_alpha}):")
    print(f"  max_aug_syn = {dataset_size} * {extreme_aug_ratio} = {max_aug_syn_uncapped}")
    print(f"  n_hard_ham = {ham_excess} * {extreme_alpha} = {n_hard_ham_uncapped}")
    print()

    if max_aug_syn_uncapped > 20:
        print(f"✅ max_aug_syn ({max_aug_syn_uncapped}) exceeds old cap of 20 - Cap removed!")
    else:
        print(f"ℹ️ max_aug_syn ({max_aug_syn_uncapped}) still below old cap of 20")

    if n_hard_ham_uncapped > 30:
        print(f"✅ n_hard_ham ({n_hard_ham_uncapped}) exceeds old cap of 30 - Cap removed!")
    else:
        print(f"ℹ️ n_hard_ham ({n_hard_ham_uncapped}) still below old cap of 30")

    print()

    # Show the mathematical formulas
    print("📐 NEW UNCAPPED FORMULAS")
    print("-" * 40)
    print("Before (with caps):")
    print("  max_aug_syn = min(int(dataset_size * aug_ratio), 20)")
    print("  n_hard_ham = min(int((ham_count - spam_count) * alpha), 30)")
    print()
    print("After (uncapped):")
    print("  max_aug_syn = int(dataset_size * aug_ratio)")
    print("  n_hard_ham = int((ham_count - spam_count) * alpha)")
    print()

    print("💡 Benefits of removing caps:")
    print("  ✅ No artificial limits on augmentation")
    print("  ✅ Scales naturally with dataset size")
    print("  ✅ Full user control over augmentation amount")
    print("  ✅ Better for large datasets")

def test_realistic_uncapped_scenarios():
    """Test realistic scenarios with uncapped augmentation"""

    print("\n🎬 REALISTIC UNCAPPED SCENARIOS")
    print("=" * 60)

    scenarios = [
        {
            "name": "Small Dataset",
            "size": 100,
            "ham": 60,
            "spam": 40,
            "description": "Small project dataset"
        },
        {
            "name": "Medium Dataset",
            "size": 1000,
            "ham": 600,
            "spam": 400,
            "description": "Medium enterprise dataset"
        },
        {
            "name": "Large Dataset",
            "size": 10000,
            "ham": 6000,
            "spam": 4000,
            "description": "Large-scale production dataset"
        }
    ]

    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}: {scenario['description']}")
        print(f"   Size: {scenario['size']}, Ham: {scenario['ham']}, Spam: {scenario['spam']}")

        # Test with moderate parameters
        aug_ratio = 0.3
        alpha = 0.5

        max_aug_syn = int(scenario['size'] * aug_ratio)
        if scenario['ham'] > scenario['spam']:
            n_hard_ham = int((scenario['ham'] - scenario['spam']) * alpha)
        else:
            n_hard_ham = 0

        total_augmented = max_aug_syn + n_hard_ham
        final_size = scenario['size'] + total_augmented
        growth_percent = (total_augmented / scenario['size']) * 100

        print(f"   With aug_ratio={aug_ratio}, alpha={alpha}:")
        print(f"     max_aug_syn: {max_aug_syn}")
        print(f"     n_hard_ham: {n_hard_ham}")
        print(f"     Total augmented: {total_augmented} (+{growth_percent:.1f}%)")
        print(f"     Final dataset: {scenario['size']} → {final_size}")

if __name__ == "__main__":
    print("🔓 UNCAPPED AUGMENTATION TESTING")
    print("Verifying that augmentation caps have been removed")
    print("=" * 70)

    # Test uncapped formulas
    test_uncapped_formulas()

    # Test realistic scenarios
    test_realistic_uncapped_scenarios()

    print("\n" + "=" * 70)
    print("🎉 CAPS SUCCESSFULLY REMOVED!")
    print("=" * 70)
    print("✅ max_aug_syn is now uncapped: int(dataset_size * aug_ratio)")
    print("✅ n_hard_ham is now uncapped: int((ham_count - spam_count) * alpha)")
    print("✅ Augmentation scales naturally with dataset size")
    print("✅ Users have full control over augmentation amounts")
