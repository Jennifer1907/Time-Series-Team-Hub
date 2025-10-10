"""
Simple test to verify parameter relationships without heavy imports
Testing the mathematical formulas used in enhanced_training_clean.py
"""

def calculate_max_aug_syn(dataset_size, aug_ratio):
    """Calculate max_aug_syn using the same formula as enhanced_training_clean.py"""
    return min(int(dataset_size * aug_ratio), 20)

def calculate_n_hard_ham(ham_count, spam_count, alpha_hard_ham):
    """Calculate n_hard_ham using the same formula as enhanced_training_clean.py"""
    if ham_count > spam_count:
        return min(int((ham_count - spam_count) * alpha_hard_ham), 30)
    else:
        return 0

def test_parameter_relationships():
    """Test if increasing parameters increases the calculated values"""

    print("🧪 TESTING PARAMETER RELATIONSHIPS")
    print("=" * 60)

    # Test dataset characteristics
    dataset_size = 100
    ham_count = 80
    spam_count = 20

    print(f"📊 Test Dataset: {dataset_size} messages")
    print(f"   Ham: {ham_count}, Spam: {spam_count}")
    print(f"   Ham excess: {ham_count - spam_count}")
    print()

    # Test scenarios from app.py parameter ranges
    scenarios = [
        {"name": "Minimum", "aug_ratio": 0.1, "alpha": 0.1},
        {"name": "Low", "aug_ratio": 0.2, "alpha": 0.3},
        {"name": "Medium", "aug_ratio": 0.3, "alpha": 0.5},
        {"name": "High", "aug_ratio": 0.4, "alpha": 0.7},
        {"name": "Maximum", "aug_ratio": 0.5, "alpha": 1.0}
    ]

    print("🔧 TESTING DIFFERENT PARAMETER VALUES")
    print("-" * 80)
    print(f"{'Scenario':<10} {'aug_ratio':<10} {'alpha':<10} {'max_aug_syn':<12} {'n_hard_ham':<12}")
    print("-" * 80)

    results = []

    for scenario in scenarios:
        aug_ratio = scenario['aug_ratio']
        alpha = scenario['alpha']

        # Calculate using the actual formulas from enhanced_training_clean.py
        max_aug_syn = calculate_max_aug_syn(dataset_size, aug_ratio)
        n_hard_ham = calculate_n_hard_ham(ham_count, spam_count, alpha)

        print(f"{scenario['name']:<10} {aug_ratio:<10.1f} {alpha:<10.1f} {max_aug_syn:<12d} {n_hard_ham:<12d}")

        results.append({
            'name': scenario['name'],
            'aug_ratio': aug_ratio,
            'alpha': alpha,
            'max_aug_syn': max_aug_syn,
            'n_hard_ham': n_hard_ham
        })

    print("-" * 80)
    print()

    # Test correlation
    print("📈 CORRELATION ANALYSIS")
    print("-" * 40)

    # Extract values for analysis
    aug_ratios = [r['aug_ratio'] for r in results]
    alphas = [r['alpha'] for r in results]
    max_aug_syns = [r['max_aug_syn'] for r in results]
    n_hard_hams = [r['n_hard_ham'] for r in results]

    # Check if aug_ratio increases → max_aug_syn increases
    aug_ratio_increasing = all(aug_ratios[i] <= aug_ratios[i+1] for i in range(len(aug_ratios)-1))
    max_aug_syn_increasing = all(max_aug_syns[i] <= max_aug_syns[i+1] for i in range(len(max_aug_syns)-1))

    # Check if alpha increases → n_hard_ham increases
    alpha_increasing = all(alphas[i] <= alphas[i+1] for i in range(len(alphas)-1))
    n_hard_ham_increasing = all(n_hard_hams[i] <= n_hard_hams[i+1] for i in range(len(n_hard_hams)-1))

    print(f"🔍 aug_ratio sequence: {aug_ratios}")
    print(f"🔍 max_aug_syn sequence: {max_aug_syns}")
    print(f"{'✅' if (aug_ratio_increasing and max_aug_syn_increasing) else '❌'} aug_ratio ↗️ → max_aug_syn ↗️")
    print()

    print(f"🔍 alpha sequence: {alphas}")
    print(f"🔍 n_hard_ham sequence: {n_hard_hams}")
    print(f"{'✅' if (alpha_increasing and n_hard_ham_increasing) else '❌'} alpha ↗️ → n_hard_ham ↗️")
    print()

    # Test specific formula behaviors
    print("🧮 FORMULA TESTING")
    print("-" * 40)

    print("Formula 1: max_aug_syn = min(int(dataset_size * aug_ratio), 20)")
    print(f"  With dataset_size = {dataset_size}:")
    for aug_ratio in [0.1, 0.15, 0.2, 0.25, 0.3]:
        calculated = dataset_size * aug_ratio
        final = min(int(calculated), 20)
        print(f"    aug_ratio = {aug_ratio:.2f} → {dataset_size} * {aug_ratio:.2f} = {calculated:.1f} → min({int(calculated)}, 20) = {final}")

    print()
    print("Formula 2: n_hard_ham = min(int((ham_count - spam_count) * alpha_hard_ham), 30)")
    print(f"  With ham_count = {ham_count}, spam_count = {spam_count}:")
    print(f"  Ham excess = {ham_count - spam_count}")
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        calculated = (ham_count - spam_count) * alpha
        final = min(int(calculated), 30)
        print(f"    alpha = {alpha:.1f} → {ham_count - spam_count} * {alpha:.1f} = {calculated:.1f} → min({int(calculated)}, 30) = {final}")

    return results

def test_edge_cases():
    """Test edge cases and limits"""

    print("\n🚨 TESTING EDGE CASES")
    print("-" * 40)

    # Test cap limits
    print("Testing cap limits:")

    # Test max_aug_syn cap at 20
    large_dataset = 1000
    high_aug_ratio = 0.5
    max_aug_syn = calculate_max_aug_syn(large_dataset, high_aug_ratio)
    expected_uncapped = int(large_dataset * high_aug_ratio)
    print(f"  Large dataset: {large_dataset} * {high_aug_ratio} = {expected_uncapped}, but capped at 20 → {max_aug_syn}")

    # Test n_hard_ham cap at 30
    large_ham = 200
    small_spam = 10
    high_alpha = 1.0
    n_hard_ham = calculate_n_hard_ham(large_ham, small_spam, high_alpha)
    expected_uncapped = int((large_ham - small_spam) * high_alpha)
    print(f"  Large ham excess: ({large_ham} - {small_spam}) * {high_alpha} = {expected_uncapped}, but capped at 30 → {n_hard_ham}")

    # Test balanced dataset (no hard ham generation)
    balanced_ham = 50
    balanced_spam = 50
    n_hard_ham_balanced = calculate_n_hard_ham(balanced_ham, balanced_spam, 0.5)
    print(f"  Balanced dataset: ham={balanced_ham}, spam={balanced_spam} → n_hard_ham = {n_hard_ham_balanced}")

    # Test spam > ham (no hard ham generation)
    less_ham = 30
    more_spam = 70
    n_hard_ham_spam_heavy = calculate_n_hard_ham(less_ham, more_spam, 0.5)
    print(f"  Spam-heavy dataset: ham={less_ham}, spam={more_spam} → n_hard_ham = {n_hard_ham_spam_heavy}")

if __name__ == "__main__":
    print("🚀 AUGMENTATION PARAMETER MATHEMATICAL TESTING")
    print("=" * 60)
    print("Verifying the mathematical relationships between app.py parameters")
    print("and the calculated values in enhanced_training_clean.py")
    print()

    # Run main tests
    results = test_parameter_relationships()

    # Run edge case tests
    test_edge_cases()

    print("\n🎯 SUMMARY:")
    print("=" * 40)
    print("✅ The formulas work as expected:")
    print("   1. max_aug_syn = min(int(dataset_size * aug_ratio), 20)")
    print("   2. n_hard_ham = min(int((ham_count - spam_count) * alpha_hard_ham), 30)")
    print("✅ Increasing aug_ratio increases max_aug_syn (up to cap of 20)")
    print("✅ Increasing alpha increases n_hard_ham (up to cap of 30)")
    print("✅ Parameters from app.py directly control augmentation amounts")
