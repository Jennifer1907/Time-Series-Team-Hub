import pandas as pd

# Check original kaggle dataset
df_orig = pd.read_csv('../data/kaggle_dataset.csv')
print(f"📊 Original Kaggle Dataset: {len(df_orig)} rows")
print(f"   Categories: {df_orig['Category'].value_counts().to_dict()}")

# Check augmented kaggle dataset
df_aug = pd.read_csv('data/kaggle_dataset_augmented.csv')
print(f"\n📊 Augmented Kaggle Dataset: {len(df_aug)} rows")
print(f"   Categories: {df_aug['Category'].value_counts().to_dict()}")

print(f"\n📈 Augmentation Results:")
print(f"   Original: {len(df_orig)} rows")
print(f"   Augmented: {len(df_aug)} rows")
print(f"   Added: {len(df_aug) - len(df_orig)} rows")

# Check individual augmentation files
try:
    df_hard_spam = pd.read_csv('data/hard_spam_generated_auto.csv')
    print(f"\n✅ Hard Spam Generated: {len(df_hard_spam)} rows")
except:
    print("\n❌ Hard Spam file not found")

try:
    df_hard_ham = pd.read_csv('data/hard_ham_generated_auto.csv')
    print(f"✅ Hard Ham Generated: {len(df_hard_ham)} rows")
except:
    print("❌ Hard Ham file not found")

# Sample some augmented data
print(f"\n📝 Sample Augmented Messages:")
print("-" * 50)
for i, row in df_aug.tail(10).iterrows():
    print(f"{row['Category']}: {row['Message'][:80]}...")
