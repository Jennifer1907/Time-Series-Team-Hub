"""
Enhanced Training Integration for Spam Model

This script provides a clean interface to use the enhanced training pipeline
with your existing spam_model.py, including data augmentation.
"""

import numpy as np
import torch
import faiss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import json
from collections import Counter
from spam_model import SpamClassifier

def enhanced_augmentation(messages, labels, aug_ratio=0.2, alpha_hard_ham=0.3):
    """
    Simplified data augmentation that works with spam_model.py
    """
    try:
        from spam_model_train import augment_dataset
        return augment_dataset(messages, labels, aug_ratio=aug_ratio, alpha=alpha_hard_ham)
    except ImportError as e:
        print(f"Warning: Could not import augmentation functions: {e}")
        print("Proceeding without data augmentation")
        return [], []
    except Exception as e:
        print(f"Warning: Augmentation failed: {e}")
        print("Proceeding without data augmentation")
        return [], []

def run_enhanced_pipeline_standalone(messages, labels, classification_language='English',
                                   test_size=0.2, use_augmentation=True,
                                   aug_ratio=0.2, alpha_hard_ham=0.3, progress_callback=None):
    """
    Standalone enhanced pipeline that integrates with spam_model.py
    """
    print("=== Enhanced Spam Classification Pipeline ===")

    if progress_callback:
        progress_callback(0.05, "Initializing classifier...")

    # 1. Initialize classifier
    classifier = SpamClassifier(classification_language=classification_language)

    if progress_callback:
        progress_callback(0.10, "Processing data...")

    # 2. Data augmentation
    original_count = len(messages)
    if use_augmentation:
        print("\n=== Data Augmentation ===")
        try:
            aug_messages, aug_labels = enhanced_augmentation(
                messages, labels, aug_ratio=aug_ratio, alpha_hard_ham=alpha_hard_ham
            )
            if aug_messages:
                messages = messages + aug_messages
                labels = labels + aug_labels
                print(f"📈 Dataset size: {original_count} → {len(messages)} (+{len(aug_messages)})")
                if progress_callback:
                    progress_callback(0.20, f"Data augmented: +{len(aug_messages)} samples")
        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            if progress_callback:
                progress_callback(0.20, "Augmentation failed, proceeding without")
    else:
        print("ℹ️ Data augmentation disabled")
        if progress_callback:
            progress_callback(0.20, "Skipping augmentation")

    if progress_callback:
        progress_callback(0.25, "Starting model training...")

    # 3. Train with enhanced dataset using existing train method
    results = classifier.train(
        messages, labels,
        test_size=test_size,
        progress_callback=lambda p, msg: progress_callback(0.25 + p * 0.65, msg) if progress_callback else None
    )

    if progress_callback:
        progress_callback(0.90, "Saving enhanced model...")

    # 4. Save enhanced model
    classifier.save_to_files()

    if progress_callback:
        progress_callback(1.0, "Enhanced training completed!")

    # 5. Return comprehensive results
    enhanced_results = {
        'classifier': classifier,
        'original_dataset_size': original_count,
        'final_dataset_size': len(messages),
        'augmentation_count': len(messages) - original_count,
        'training_results': results,
        'model_info': classifier.model_info,
        'best_alpha': classifier.best_alpha,
        'class_weights': classifier.class_weights
    }

    print(f"\n✅ Enhanced Training Summary:")
    print(f"  Original dataset: {original_count} samples")
    print(f"  Final dataset: {len(messages)} samples")
    print(f"  Augmented samples: {len(messages) - original_count}")
    print(f"  Best alpha: {classifier.best_alpha:.3f}")
    if hasattr(classifier, 'model_info') and 'accuracy_results' in classifier.model_info:
        best_acc = max(classifier.model_info['accuracy_results'].values())
        print(f"  Best accuracy: {best_acc:.1%}")

    return enhanced_results

def add_enhanced_training_to_classifier():
    """
    Monkey patch to add enhanced training method to existing SpamClassifier
    """
    def train_enhanced(self, messages, labels, test_size=0.2, use_augmentation=True,
                      aug_ratio=0.2, alpha_hard_ham=0.3, progress_callback=None):
        """Enhanced training with data augmentation"""

        if progress_callback:
            progress_callback(0.05, "Starting enhanced training...")

        # Data augmentation
        original_count = len(messages)
        if use_augmentation:
            if progress_callback:
                progress_callback(0.10, "Augmenting dataset...")
            try:
                aug_messages, aug_labels = enhanced_augmentation(
                    messages, labels, aug_ratio=aug_ratio, alpha_hard_ham=alpha_hard_ham
                )
                if aug_messages:
                    messages = messages + aug_messages
                    labels = labels + aug_labels
                    print(f"📈 Dataset augmented: {original_count} → {len(messages)}")
                    if progress_callback:
                        progress_callback(0.20, f"Added {len(aug_messages)} augmented samples")
            except Exception as e:
                print(f"Augmentation failed: {e}, continuing without augmentation")
                if progress_callback:
                    progress_callback(0.20, "Augmentation failed, proceeding")

        if progress_callback:
            progress_callback(0.25, "Training with enhanced dataset...")

        # Continue with normal training but with adjusted progress
        return self.train(
            messages, labels,
            test_size=test_size,
            progress_callback=lambda p, msg: progress_callback(0.25 + p * 0.75, msg) if progress_callback else None
        )

    # Add the method to the class
    SpamClassifier.train_enhanced = train_enhanced
    print("✅ Enhanced training method added to SpamClassifier")

# Auto-patch the class when this module is imported
add_enhanced_training_to_classifier()

# Example usage functions
def main_enhanced_training():
    """Example of how to use enhanced training"""

    print("=== Enhanced Training Example ===")

    # Initialize classifier
    classifier = SpamClassifier(classification_language='English')

    # Load data
    print("Loading dataset...")
    messages, labels = classifier.load_dataset(source='gdrive')
    print(f"Loaded {len(messages)} messages")

    # Run enhanced training (now available as a method)
    print("Starting enhanced training...")
    results = classifier.train_enhanced(
        messages, labels,
        use_augmentation=True,
        aug_ratio=0.2,
        alpha_hard_ham=0.3
    )

    print("Enhanced training completed!")
    print(f"Best alpha: {results['best_alpha']:.3f}")
    if 'accuracy_results' in results:
        best_acc = max(results['accuracy_results'].values())
        print(f"Best accuracy: {best_acc:.1%}")

    return classifier, results

if __name__ == "__main__":
    # Run example
    classifier, results = main_enhanced_training()
    print("Example completed successfully!")
