"""
Enhanced Training Integration for Spam Model

This script provides a clean interface to use enhanced training pipeline
with data augmentation for the spam detection model.
"""

import numpy as np
import torch
import faiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import json
from collections import Counter
import random
import nltk
import warnings
import csv

warnings.filterwarnings('ignore')

# Try to import wordnet for augmentation
try:
    from nltk.corpus import wordnet
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    WORDNET_AVAILABLE = True
except:
    print("⚠️ NLTK WordNet not available, synonym replacement disabled")
    WORDNET_AVAILABLE = False

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def get_hard_ham_phrase_groups():
    """Get predefined hard ham phrase groups for augmentation"""
    financial_phrases = [
        "I got $100 cashback yesterday", "The bank refunded me $200 already",
        "I earned $150/day last month from freelancing", "Approved for $500 loan finally"
    ]

    promotion_phrases = [
        "I bought one and got one free, legit deal", "Flash sale 80% off, I already ordered",
        "Exclusive deal worked for me, saved a lot", "Hot sale 2 hours ago, crazy cheap"
    ]

    lottery_phrases = [
        "I actually won a $1000 voucher at the mall", "I got a free iPhone from the lucky draw",
        "Claimed my $500 Amazon voucher legit", "Won a prize, just showed my ticket"
    ]

    scam_alert_phrases = [
        "I got unusual login alert, but it was me", "Reset my password after warning, fine now",
        "Got security update mail, confirmed it's real", "Payment failed once, updated and ok now"
    ]

    return [financial_phrases, promotion_phrases, lottery_phrases, scam_alert_phrases]

def generate_hard_ham(ham_texts, n=30):
    """Generate hard ham examples that look like spam but are legitimate"""
    if not ham_texts or n <= 0:
        return []

    hard_ham_phrase_groups = get_hard_ham_phrase_groups()
    hard_ham = []

    for _ in range(min(n, len(ham_texts))):
        try:
            base = random.choice(ham_texts)
            insert_group = random.choice(hard_ham_phrase_groups)
            insert = random.choice(insert_group)

            if random.random() > 0.5:
                hard_ham.append(f"{base}, btw {insert}.")
            else:
                hard_ham.append(f"{insert}. {base}")
        except Exception:
            continue

    return hard_ham

def synonym_replacement(text, n=1):
    """Replace words with synonyms using WordNet"""
    if not WORDNET_AVAILABLE:
        return text

    try:
        if not isinstance(text, str):
            text = str(text)

        if not text or not text.strip():
            return text

        words = text.split()
        new_words = words.copy()

        # Filter words that have synonyms in WordNet
        candidates = []
        for w in words:
            try:
                if wordnet.synsets(w):
                    candidates.append(w)
            except:
                continue

        if not candidates:
            return text

        random.shuffle(candidates)
        replaced_count = 0

        for random_word in candidates[:n]:
            try:
                synonyms = wordnet.synsets(random_word)
                if synonyms and synonyms[0].lemmas():
                    synonym = synonyms[0].lemmas()[0].name().replace('_', ' ')
                    if synonym.lower() != random_word.lower():
                        new_words = [synonym if w == random_word else w for w in new_words]
                        replaced_count += 1
            except:
                continue

        return " ".join(new_words)

    except Exception:
        return str(text) if text else ""

def enhanced_augmentation(messages, labels, aug_ratio=0.1, alpha_hard_ham=0.2):
    """
    Simplified data augmentation for spam detection
    """
    print("=== Data Augmentation ===")

    augmented_messages = []
    augmented_labels = []

    # Ensure inputs are lists
    if not isinstance(messages, list):
        messages = list(messages)
    if not isinstance(labels, list):
        labels = list(labels)

    # Convert all messages to strings
    messages = [str(msg) for msg in messages]

    # Count current distribution
    ham_count = labels.count('ham')
    spam_count = labels.count('spam')
    print(f"📊 Original dataset: Ham={ham_count}, Spam={spam_count}")

    # 1. Hard Ham Generation (only if ham > spam)
    if ham_count > spam_count:
        ham_messages = [msg for msg, label in zip(messages, labels) if label == 'ham']
        n_hard_ham = int((ham_count - spam_count) * alpha_hard_ham)

        if n_hard_ham > 0 and ham_messages:
            print(f"🎯 Generating {n_hard_ham} hard ham examples...")
            hard_ham_generated = generate_hard_ham(ham_messages, n=n_hard_ham)

            if hard_ham_generated:
                augmented_messages.extend(hard_ham_generated)
                augmented_labels.extend(['ham'] * len(hard_ham_generated))
                print(f"✅ Generated {len(hard_ham_generated)} hard ham examples")

    # 2. Synonym Replacement (unlimited)
    max_aug_syn = int(len(messages) * aug_ratio)
    print(f"🎯 Generating up to {max_aug_syn} synonym replacement examples...")

    syn_count = 0
    for msg, label in zip(messages, labels):
        if syn_count >= max_aug_syn:
            break

        if random.random() > 0.9:  # 10% chance
            try:
                aug_msg = synonym_replacement(msg, n=1)

                if (aug_msg != msg and
                    len(aug_msg.strip()) > 0 and
                    len(aug_msg.split()) >= 2):

                    augmented_messages.append(aug_msg)
                    augmented_labels.append(label)
                    syn_count += 1

            except Exception:
                continue

    print(f"✅ Generated {syn_count} synonym replacement examples")
    print(f"✅ Total augmented: {len(augmented_messages)} examples")

    return augmented_messages, augmented_labels

def run_enhanced_pipeline(classifier_instance, messages, labels, test_size=0.2, use_augmentation=True, aug_ratio=0.1, alpha_hard_ham=0.2):
    """
    Enhanced pipeline that integrates with SpamClassifier instance
    """
    print("=== Enhanced Spam Classification Pipeline ===")
    print(f"🎛️ Using parameters: aug_ratio={aug_ratio}, alpha_hard_ham={alpha_hard_ham}")

    original_count = len(messages)

    # Apply augmentation if enabled
    if use_augmentation:
        try:
            augmented_messages, augmented_labels = enhanced_augmentation(
                messages, labels,
                aug_ratio=aug_ratio,
                alpha_hard_ham=alpha_hard_ham
            )

            print(f'📊 Augmented Message Preview: {augmented_messages[:5], augmented_labels[:5]}')

            # with open('augment_data.csv', 'w') as f:
            #     writer = csv.writer(f)
            #     field_name = ['Category', 'Message']
            #     writer = csv.DictWriter(f, fieldnames=field_name)

            if augmented_messages:
                messages = list(messages) + augmented_messages
                labels = list(labels) + augmented_labels
                print(f"📈 Dataset size: {original_count} → {len(messages)} (+{len(augmented_messages)})")
            else:
                print("ℹ️ No augmented data generated")
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            print("ℹ️ Continuing with original data...")
    else:
        print("ℹ️ Data augmentation disabled")

    # Train the model using the classifier's train method
    print("\n=== Training Enhanced Model ===")
    training_results = classifier_instance.train(messages, labels, test_size=test_size)

    # Create comprehensive results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": classifier_instance.model_name,
        "original_dataset_size": original_count,
        "final_dataset_size": len(messages),
        "augmentation_enabled": use_augmentation,
        "test_size": test_size,
        "training_results": convert_numpy_types(training_results)
    }

    # Update the classifier's model_info to include augmentation information
    if hasattr(classifier_instance, 'model_info') and classifier_instance.model_info:
        classifier_instance.model_info.update({
            "original_dataset_size": original_count,
            "final_dataset_size": len(messages),
            "augmentation_enabled": use_augmentation,
            "augmentation_count": len(messages) - original_count
        })

    # Save enhanced results
    with open("enhanced_training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Enhanced training completed!")
    print(f"📊 Dataset: {original_count} → {len(messages)} samples")
    print(f"📈 Best Alpha: {training_results.get('best_alpha', 'N/A')}")
    print(f"💾 Results saved to enhanced_training_results.json")

    return {
        "classifier": classifier_instance,
        "results": results,
        "enhanced_messages": messages,
        "enhanced_labels": labels
    }

def add_enhanced_method_to_classifier():
    """
    Dynamically add the run_enhanced_pipeline method to SpamClassifier class
    """
    try:
        from spam_model import SpamClassifier

        def train_enhanced(self, messages, labels, test_size=0.2, use_augmentation=True, aug_ratio=0.1, alpha_hard_ham=0.2):
            """Enhanced training method with data augmentation"""
            return run_enhanced_pipeline(self, messages, labels, test_size, use_augmentation, aug_ratio, alpha_hard_ham)

        # Add the method to the class
        SpamClassifier.run_enhanced_pipeline = train_enhanced
        print("✅ Enhanced training method added to SpamClassifier")
        return SpamClassifier

    except ImportError as e:
        print(f"❌ Could not import SpamClassifier: {e}")
        return None
    except Exception as e:
        print(f"❌ Error adding enhanced method: {e}")
        return None

if __name__ == "__main__":
    print("=== Enhanced Training Module for Spam Detection ===")
    print()

    # Try to add enhanced method to SpamClassifier
    SpamClassifier = add_enhanced_method_to_classifier()

    if SpamClassifier:
        try:
            print("🚀 Testing Enhanced Pipeline...")

            # Initialize classifier
            classifier = SpamClassifier(classification_language='English')

            # Load dataset
            print("📂 Loading dataset...")
            messages, labels = classifier.load_dataset(source='kaggle')

            # Use a smaller subset for testing
            if len(messages) > 200:
                print(f"📊 Using subset of 200 samples for testing...")
                messages = messages[:200]
                labels = labels[:200]

            # Run enhanced pipeline
            print("\n🔥 Running Enhanced Pipeline...")
            results = classifier.run_enhanced_pipeline(
                messages, labels,
                test_size=0.2,
                use_augmentation=True
            )

            print("\n🎉 Enhanced training completed successfully!")
            print("\n📋 Usage Example:")
            print("```python")
            print("from enhanced_training import add_enhanced_method_to_classifier")
            print("from spam_model import SpamClassifier")
            print()
            print("# Add enhanced method to classifier")
            print("add_enhanced_method_to_classifier()")
            print()
            print("# Initialize and use classifier")
            print("classifier = SpamClassifier()")
            print("messages, labels = classifier.load_dataset(source='kaggle')")
            print("results = classifier.run_enhanced_pipeline(messages, labels, use_augmentation=True)")
            print("```")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            print("💡 You can still use the enhanced_augmentation function manually")

    else:
        print("💡 Manual usage:")
        print("```python")
        print("from enhanced_training import enhanced_augmentation, run_enhanced_pipeline")
        print("augmented_msgs, augmented_lbls = enhanced_augmentation(messages, labels)")
        print("```")
