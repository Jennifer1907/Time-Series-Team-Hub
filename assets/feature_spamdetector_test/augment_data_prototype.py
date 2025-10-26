# ==============================
# Class sinh dữ liệu tinh vi để augmentation
# ==============================
import kagglehub
import pandas as pd
import random
import requests
import subprocess
import os
import gdown
from spam_model import SpamClassifier

# Download NLTK data if needed
try:
    import nltk
    from nltk.corpus import wordnet
    # Try to access wordnet, download if not available
    try:
        wordnet.synsets('test')
        WORDNET_AVAILABLE = True
    except:
        print("📦 Downloading NLTK wordnet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        try:
            wordnet.synsets('test')
            WORDNET_AVAILABLE = True
        except:
            WORDNET_AVAILABLE = False
            print("⚠️ Failed to load NLTK wordnet")
except ImportError:
    print("⚠️ NLTK not available, synonym replacement will be limited")
    wordnet = None
    WORDNET_AVAILABLE = False


class HardExampleGenerator:
    def __init__(self, dataset_path, alpha_spam=0.5, alpha_ham=0.3, use_llm_phrases=False):
        """
        Args:
            dataset_path (str): đường dẫn file CSV chứa cột 'Message' và 'Category'
            alpha_spam (float): tỷ lệ nhân bản spam khi augment
            alpha_ham (float): tỷ lệ nhân bản ham khi augment
            use_llm_phrases (bool): nếu True thì chờ load LLM phrases sau bằng load_llm_phrases()
        """
        self.dataset_path = dataset_path
        self.alpha_spam = alpha_spam
        self.alpha_ham = alpha_ham
        self.df = pd.read_csv(dataset_path)
        self.TOGETHER_AI_API_KEY = "a4910347ea0b1f86be877cd19899dd0bd3f855487a0b80eb611a64c0abf7a782"

        # Nếu chưa có LLM phrases thì dùng cụm mặc định
        if not use_llm_phrases:
            self.spam_groups = self._init_spam_phrases()
            self.ham_groups = self._init_ham_phrases()
        else:
            # Khởi tạo rỗng, sau sẽ gán bằng load_llm_phrases()
            self.spam_groups = []
            self.ham_groups = []


    # Dùng cho cách 2 (có thể lấy từ LLM bên ngoài xịn hơn)
    def _init_spam_phrases(self):
        # Các cụm spam tinh vi (giống file gốc)
        # ----- 7 nhóm dấu hiệu spam -----
        financial_phrases = [
            "you get $100 back", "they refund $200 instantly",
            "limited $50 bonus for early registration", "earn $150/day remote work",
            "approved for a $500 credit", "quick $300 refund if you confirm",
            "they give $250 cashback if you check in early",
            "your account gets $100 instantly after confirmation",
            "instant $400 transfer if you reply YES today",
            "exclusive $600 grant approved for you"
        ]

        promotion_phrases = [
            "limited time offer ends tonight", "buy one get one free today only",
            "exclusive deal just for you", "hot sale up to 80% off",
            "flash sale starting in 2 hours", "new collection, free shipping worldwide",
            "best price guaranteed for early birds", "special discount coupon for first 100 buyers",
            "reserve now and get extra 20% off", "only 3 items left, order now!"
        ]

        lottery_phrases = [
            "congratulations! you’ve won a $1000 gift card", "you are selected to receive a free iPhone",
            "claim your $500 Amazon voucher now", "winner! reply to confirm your prize",
            "spin the wheel to win exciting gifts", "lucky draw winner – act fast",
            "redeem your exclusive prize today", "final reminder: unclaimed reward waiting",
            "instant gift unlocked, tap to get", "biggest jackpot giveaway this week"
        ]

        scam_alert_phrases = [
            "your account will be suspended unless verified", "unusual login detected, reset password now",
            "security update required immediately", "urgent: payment failed, update details now",
            "verify your identity to avoid account closure", "your Netflix subscription is on hold, confirm payment",
            "important: unauthorized activity detected", "bank alert: confirm transaction or account locked",
            "last warning: confirm within 24 hours", "emergency: suspicious access blocked, verify"
        ]

        call_to_action_phrases = [
            "click here to confirm", "reply YES to activate bonus",
            "register before midnight and win", "tap now to claim your reward",
            "sign up today, limited seats", "confirm immediately to proceed",
            "act fast, offer expires soon", "verify email to continue",
            "download the app and get free points", "complete payment within 12 hours"
        ]

        social_engineering_phrases = [
            "hey grandma, i need $500 for hospital bills", "hi mom, send money asap, phone broke",
            "boss asked me to buy 3 gift cards urgently", "john, can you transfer $300 now, emergency",
            "it’s me, your cousin, stuck abroad, need help", "friend, please help me with $200 loan",
            "hi, i lost my wallet, send $150 to this account", "urgent! i can’t talk now, send cash fast",
            "help me pay this fine, will return tomorrow", "sister, please pay $400 for my surgery"
        ]

        obfuscated_phrases = [
            "Cl!ck h3re t0 w1n fr€e iPh0ne", "G€t y0ur r3fund n0w!!!",
            "L!mited 0ff3r: Fr33 $$$ r3ward", "C@shb@ck av@il@ble t0d@y",
            "W!n b!g pr!ze, act f@st", "Cl@im y0ur 100% b0nus",
            "Fr33 g!ft w!th 0rder", "Up t0 $5000 r3fund @pprov3d",
            "R3ply N0W t0 r3c3ive $$$", "Urg3nt!!! C0nfirm d3tails 1mm3di@tely"
        ]

        # Gom các nhóm vào 1 danh sách
        spam_phrase_groups = [
            financial_phrases, promotion_phrases, lottery_phrases,
            scam_alert_phrases, call_to_action_phrases,
            social_engineering_phrases, obfuscated_phrases
        ]
        return spam_phrase_groups

    # Dùng cho cách 2 (có thể lấy từ LLM bên ngoài xịn hơn)
    def _init_ham_phrases(self):
        # Các cụm ham dễ gây hiểu nhầm
        # ----- 7 nhóm cụm dễ gây hiểu nhầm thành spam (giống spam phrases) -----
        financial_phrases = [
            "I got $100 cashback yesterday", "The bank refunded me $200 already",
            "I earned $150/day last month from freelancing", "Approved for $500 loan finally",
            "Got quick $300 refund after confirmation", "The store gave me $250 cashback",
            "My account got $100 instantly after confirming", "I received instant $400 transfer today",
            "They sent me exclusive $600 grant, lol", "Netflix actually gave me 3 months free"
        ]

        promotion_phrases = [
            "I bought one and got one free, legit deal", "Flash sale 80% off, I already ordered",
            "Exclusive deal worked for me, saved a lot", "Hot sale 2 hours ago, crazy cheap",
            "New collection free shipping, I tried it", "Best price ever for members",
            "Got special coupon, it worked!", "Reserved early and saved 20%",
            "Only 3 items left when I bought mine", "Order now, it’s real not fake"
        ]

        lottery_phrases = [
            "I actually won a $1000 voucher at the mall", "I got a free iPhone from the lucky draw",
            "Claimed my $500 Amazon voucher legit", "Won a prize, just showed my ticket",
            "Spun the wheel at the fair and got gifts", "Lucky draw worked for me today",
            "Redeemed my exclusive prize at the shop", "They reminded me to collect my reward",
            "Gift unlocked at the event, so fun", "Jackpot giveaway, real not scam"
        ]

        scam_alert_phrases = [
            "I got unusual login alert, but it was me", "Reset my password after warning, fine now",
            "Got security update mail, confirmed it’s real", "Payment failed once, updated and ok now",
            "Had to verify identity, bank confirmed legit", "Netflix on hold but paid, no issue",
            "Bank asked to confirm transaction, was me", "Warning mail yesterday, false alarm",
            "Confirmed within 24h, all safe", "Suspicious access blocked, just me traveling"
        ]

        call_to_action_phrases = [
            "I clicked to confirm and it worked", "Replied YES, bonus legit",
            "Registered before midnight, no scam", "Tapped link, claimed reward legit",
            "Signed up today, limited seat real", "Confirmed immediately, nothing shady",
            "Acted fast, got discount legit", "Verified email, safe and done",
            "Downloaded app, free points real", "Paid within 12 hours, successful"
        ]

        social_engineering_phrases = [
            "Mom, don’t worry I sent you $500 hospital bill already", "Hi mom, phone broke but friend helped",
            "Boss asked me to buy gift cards for office, already did", "John, I transferred $300, check it",
            "Cousin stuck abroad, we sent help", "Friend lent me $200 last week, repaid",
            "Lost wallet but someone returned $150", "Urgent cash request yesterday, sorted now",
            "Helped pay fine, friend returned", "Sister’s surgery done, paid $400 legit"
        ]

        obfuscated_phrases = [
            "Clicked h3re to win fr€e gift, real promo", "Got r3fund n0w!!! 100% legit",
            "Fr33 reward worked, tried it", "C@shb@ck real, used today",
            "Won prize real, not spam", "Cl@imed b0nus myself, safe",
            "Gift order legit, no scam", "Refund approved @ bank, no issue",
            "Replied N0W got $$$ legit", "Urg3nt confirm done, real bank"
        ]

        hard_ham_phrase_groups = [
            financial_phrases, promotion_phrases, lottery_phrases,
            scam_alert_phrases, call_to_action_phrases,
            social_engineering_phrases, obfuscated_phrases
        ]
        return hard_ham_phrase_groups

    # Dùng cho cách 1
    def generate_like_spam_ham(self, label='spam', n_per_group=10, model="mistralai/Mixtral-8x7B-Instruct-v0.1", group=None):
        """
        Sinh các câu spam/ham tinh vi mô phỏng người dùng, chia theo 7 nhóm phổ biến (70 câu total).

        Args:
            label (str): 'spam' hoặc 'ham'
            n_per_group (int): số câu trên mỗi nhóm
            api_key (str): Together.ai API key
            model (str): Model ID (Mixtral, LLaMA3,...)
            group (str or None): nếu chỉ muốn sinh 1 nhóm, chọn từ:
                'financial', 'promotion', 'lottery', 'scam_alert',
                'call_to_action', 'social_engineering', 'obfuscated'

        Returns:
            List[str]: Danh sách câu được sinh
        """
        api_key = self.TOGETHER_AI_API_KEY

        if api_key is None:
            raise ValueError("❌ Cần cung cấp API key Together.ai.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        group_prompts = {
            "financial": {
                "spam": "Generate realistic user-style spam messages that pretend to offer cashback, refunds, or financial rewards, but are actually deceptive.",
                "ham": "Generate legitimate human messages that mention refunds, cashback, or money transfers in real-life, harmless contexts."
            },
            "promotion": {
                "spam": "Generate spammy messages that appear friendly but are disguised promotions, sales, or limited-time offers.",
                "ham": "Generate genuine user messages that talk about real promotions or sales they used, sounding casual and truthful."
            },
            "lottery": {
                "spam": "Generate scam-like messages that claim the user won a lottery, prize, or giveaway — but in a deceptive, subtle tone.",
                "ham": "Generate honest messages where users talk about actually winning something in real life — malls, fairs, etc."
            },
            "scam_alert": {
                "spam": "Generate deceptive user-style spam about account alerts, security warnings, or password issues to trick the recipient.",
                "ham": "Generate real user messages where people talk about security alerts or login issues they experienced, in normal tone."
            },
            "call_to_action": {
                "spam": "Write spam messages with subtle calls to action like 'click here', 'register', or 'confirm' hidden in casual tone.",
                "ham": "Write normal human messages that mention clicking links or confirming actions, but are not spam."
            },
            "social_engineering": {
                "spam": "Generate spam messages that use fake urgency or personal relationships (e.g., 'Mom', 'Boss', 'Friend') to request money.",
                "ham": "Generate real messages from people who had real emergencies or money transfers, in personal tone."
            },
            "obfuscated": {
                "spam": "Write spam messages that use obfuscated text like '$$$', 'Fr33', 'Cl!ck', to bypass filters but sound human.",
                "ham": "Write real human messages that coincidentally use symbols or strange formats, but are not spam."
            }
        }

        selected_groups = [group] if group else list(group_prompts.keys())
        all_outputs = []

        for g in selected_groups:
            system_prompt = group_prompts[g][label]
            full_prompt = f"{system_prompt}\nGenerate {n_per_group} examples. Output only the messages, one per line."

            payload = {
                "model": model,
                "prompt": full_prompt,
                "max_tokens": 1000,
                "temperature": 0.9,
                "top_p": 0.95
            }

            print(f"📡 Generating {label.upper()} – Group: {g} ...")

            response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)

            if response.ok:
                raw_output = response.json()["choices"][0]["text"].strip()
                lines = [line.strip("-•* ") for line in raw_output.splitlines() if line.strip()]
                all_outputs.extend(lines)
            else:
                raise RuntimeError(f"❌ API error @group {g}: {response.status_code} - {response.text}")

        return all_outputs

    # Dùng cho cách 1
    def load_llm_phrases(self, spam_list, ham_list, group_size=10):
        """
        Từ danh sách 70 câu spam + 70 câu ham, chia thành 7 nhóm (mỗi nhóm 10 câu).
        Dùng thay cho _init_spam_phrases() và _init_ham_phrases()

        Args:
            spam_list (list[str]): Danh sách 70 câu spam từ LLM
            ham_list (list[str]): Danh sách 70 câu ham từ LLM
            group_size (int): Số câu mỗi nhóm (mặc định 10)

        Tác dụng:
            Gán trực tiếp vào self.spam_groups và self.ham_groups
        """
        #assert len(spam_list) == len(ham_list) == 70, "❌ Cần đúng 70 câu mỗi loại để chia nhóm."
        spam_list = spam_list[:70]
        ham_list = ham_list[:70]
        self.spam_groups = [spam_list[i:i+group_size] for i in range(0, 70, group_size)]
        self.ham_groups = [ham_list[i:i+group_size] for i in range(0, 70, group_size)]
        print("✅ Đã load 140 câu LLM và chia thành 7 nhóm spam/ham.")
        return self.spam_groups, self.ham_groups

    def _generate_sentences(self, base_texts, phrase_groups, n):
        results = []
        for _ in range(n):
            base = random.choice(base_texts)
            insert = random.choice(random.choice(phrase_groups))
            sentence = f"{insert}. {base}" if random.random() < 0.5 else f"{base}, btw {insert}."
            results.append(sentence)
        return results

    def generate_hard_spam(self, output_path="/content/hard_spam_generated_auto.csv"):
        num_ham = self.df[self.df["Category"] == "ham"].shape[0]
        num_spam = self.df[self.df["Category"] == "spam"].shape[0]
        if num_spam >= num_ham:
            print("✅ Spam đã đủ, không sinh thêm.")
            return []
        n_generate = int((num_ham - num_spam) * self.alpha_spam)
        base_texts = self.df[self.df["Category"] == "ham"]["Message"].sample(n=n_generate, random_state=42).tolist()
        generated = self._generate_sentences(base_texts, self.spam_groups, n_generate)
        pd.DataFrame({"Category": ["spam"] * n_generate, "Message": generated}).to_csv(output_path, index=False)
        print(f"✅ Sinh {n_generate} hard spam -> {output_path}")
        return generated

    def generate_hard_ham(self, output_path="/content/hard_ham_generated_auto.csv"):
        num_ham = self.df[self.df["Category"] == "ham"].shape[0]
        num_spam = self.df[self.df["Category"] == "spam"].shape[0]
        if num_ham >= num_spam:
            n_generate = int((num_ham - num_spam) * self.alpha_ham)
            base_texts = self.df[self.df["Category"] == "ham"]["Message"].sample(n=n_generate, random_state=42).tolist()
            generated = self._generate_sentences(base_texts, self.ham_groups, n_generate)
            pd.DataFrame({"Category": ["ham"] * n_generate, "Message": generated}).to_csv(output_path, index=False)
            print(f"✅ Sinh {n_generate} hard ham -> {output_path}")
            return generated
        else:
            print("✅ Ham đã đủ, không cần sinh thêm.")
            return []

    def generate_synonym_replacement(self, messages, labels, aug_ratio=0.2):
        MAX_AUG = int(len(messages) * aug_ratio)
        augmented_messages, augmented_labels = [], []
        print(f"✅ Synonym Replacement: sinh tối đa {MAX_AUG} câu.")
        for msg, label in zip(messages, labels):
            if len(augmented_messages) >= MAX_AUG:
                break
            if random.random() > 0.8:
                aug_msg = self.synonym_replacement(msg)
                if aug_msg != msg:
                    augmented_messages.append(aug_msg)
                    augmented_labels.append(label)
        print(f"✅ Đã sinh {len(augmented_messages)} câu augmented thực tế.")
        return augmented_messages, augmented_labels

    def synonym_replacement(self, text, n=1):
        """Replace words with synonyms using WordNet (if available)"""
        if not WORDNET_AVAILABLE or wordnet is None:
            return text  # Return original text if wordnet is not available

        words = text.split()
        new_words = words.copy()

        try:
            # Check which words have synonyms
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

            for random_word in candidates:
                try:
                    synsets = wordnet.synsets(random_word)
                    if synsets and len(synsets) > 0:
                        lemmas = synsets[0].lemmas()
                        if lemmas and len(lemmas) > 0:
                            synonym = lemmas[0].name().replace('_', ' ')
                            if synonym.lower() != random_word.lower():
                                new_words = [synonym if w == random_word else w for w in new_words]
                                replaced_count += 1
                except:
                    continue  # Skip this word if error occurs

                if replaced_count >= n:
                    break

            return " ".join(new_words)
        except:
            return text  # Return original if any error occurs

    # Tự động sinh test case
    def generate_user_like_spam_ham(self, label='spam', n=10, api_key=None, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """
        Sinh ra các câu spam/ham giống như tin nhắn từ người dùng thật có nội dung hỏi hoặc trò chuyện.

        Args:
            label (str): 'spam' hoặc 'ham'.
            n (int): Số lượng cần sinh.
            api_key (str): Together.ai API key.
            model (str): Model ID (Mixtral/Mistral/LLaMA3...)

        Returns:
            List[str]: Danh sách tin nhắn được sinh ra.
        """

        if api_key is None:
            raise ValueError("❌ Bạn cần cung cấp Together.ai API key.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        prompt_template = {
            "spam": (
                "You are writing deceptive user messages that look like innocent questions, but are actually subtle spam.\n"
                "Generate realistic user messages (in casual style) that include spam signals, but sound like real human questions or messages.\n"
                f"Generate {n} such examples. Output only the messages, one per line."
            ),
            "ham": (
                "You are writing user messages that look like spam at first, but are actually legitimate, honest messages.\n"
                "Generate realistic messages where a user might mention cashback, refund, login alerts, etc. but in a real, harmless context.\n"
                f"Generate {n} such examples. Output only the messages, one per line."
            )
        }

        prompt = prompt_template[label]

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.9,
            "top_p": 0.95,
            "stop": None
        }

        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)

        if response.ok:
            raw_output = response.json()["choices"][0]["text"].strip()
            # Tách các dòng nếu có xuống dòng
            return [line.strip("-•* ") for line in raw_output.splitlines() if line.strip()]
        else:
            raise RuntimeError(f"Lỗi khi gọi Together API: {response.status_code} - {response.text}")


#? RUN ENHANCED PIPELINE EXAMPLE
def run_enhanced_pipeline(messages, labels, source_type, dataset_path=None, api_key=None, test_size=0.2, use_augmentation=True, augment_for_testing=True):
    """
    Run the complete enhanced spam classification pipeline

    Args:
        messages (list): Original messages
        labels (list): Original labels
        source_type (str): 'kaggle' or 'gdrive'
        dataset_path (str): Path to original dataset for augmentation
        api_key (str): Together AI API key
        test_size (float): Test split ratio
        use_augmentation (bool): Whether to use augmentation
        augment_for_testing (bool): Use minimal augmentation for testing

    Returns:
        dict: Pipeline results with augmented data
    """
    print("=== Enhanced Spam Classification Pipeline ===")
    print(f"📊 Original dataset: {len(messages)} messages")
    print(f"📁 Source type: {source_type}")

    # Create data folder if not exists
    os.makedirs("data", exist_ok=True)

    # Save original data to CSV first
    original_df = pd.DataFrame({"Category": labels, "Message": messages})
    if source_type == 'kaggle':
        original_file = "data/kaggle_dataset.csv"
    else:
        original_file = "data/gdrive_dataset.csv"

    original_df.to_csv(original_file, index=False)
    print(f"💾 Saved original data to: {original_file}")

    # 1. DATA AUGMENTATION
    if use_augmentation:
        print("\n=== Data Augmentation ===")

        try:
            # 1.1. Choose augmentation method (minimal for testing)
            if augment_for_testing:
                print("🧪 Testing mode: Using minimal augmentation")
                aug_mode = "2"  # Use built-in phrases to save API calls
            else:
                print("Chọn cách data augmentation:")
                print("1. Sinh câu tinh vi bằng LLM (dùng API Together.ai)")
                print("2. Dùng cụm câu có sẵn trong code (không cần mạng/API)")
                aug_mode = input("👉 Nhập 1 hoặc 2: ").strip()

            use_llm = aug_mode == "1"

            # Use the saved original file as dataset_path for augmentation
            dataset_path = original_file

            gen = HardExampleGenerator(
                dataset_path=dataset_path,
                alpha_spam=0.3 if augment_for_testing else 1.0,  # Smaller alpha for testing
                alpha_ham=0.2 if augment_for_testing else 0.3,   # Smaller alpha for testing
                use_llm_phrases=use_llm
            )

            if use_llm:
                if api_key:
                    print("🤖 Generating with LLM (minimal for testing)...")
                    # Use smaller numbers for testing to save API calls
                    n_per_group = 1 if augment_for_testing else 2
                    llm_spam = gen.generate_like_spam_ham(label='spam', n_per_group=n_per_group)
                    llm_ham = gen.generate_like_spam_ham(label='ham', n_per_group=n_per_group)
                    gen.load_llm_phrases(spam_list=llm_spam, ham_list=llm_ham)
                else:
                    print("⚠️ Không có API key. Sử dụng cụm có sẵn.")
                    gen.spam_groups = gen._init_spam_phrases()
                    gen.ham_groups = gen._init_ham_phrases()
            else:
                print("ℹ️ Sử dụng cụm đã được hardcode trong class.")

            # 1.2. Generate augmented data (save to data folder)
            hard_spam_path = "data/hard_spam_generated_auto.csv"
            hard_ham_path = "data/hard_ham_generated_auto.csv"

            hard_spam_generated = gen.generate_hard_spam(hard_spam_path)
            hard_ham_generated = gen.generate_hard_ham(hard_ham_path)

            # Generate synonym replacement with smaller ratio for testing
            syn_ratio = 0.1 if augment_for_testing else 0.2
            augmented_messages, augmented_labels = gen.generate_synonym_replacement(messages, labels, aug_ratio=syn_ratio)

            # 1.3. Merge all data into one DataFrame
            df_base = gen.df

            # Read generated files if they exist
            df_hard_spam = pd.DataFrame()
            df_hard_ham = pd.DataFrame()

            if os.path.exists(hard_spam_path):
                df_hard_spam = pd.read_csv(hard_spam_path)
                print(f"✅ Hard spam generated: {len(df_hard_spam)} samples")

            if os.path.exists(hard_ham_path):
                df_hard_ham = pd.read_csv(hard_ham_path)
                print(f"✅ Hard ham generated: {len(df_hard_ham)} samples")

            df_synonym = pd.DataFrame({"Category": augmented_labels, "Message": augmented_messages})
            print(f"✅ Synonym replacement: {len(df_synonym)} samples")

            # Combine all dataframes
            dfs_to_concat = [df_base]
            if not df_hard_spam.empty:
                dfs_to_concat.append(df_hard_spam)
            if not df_hard_ham.empty:
                dfs_to_concat.append(df_hard_ham)
            if not df_synonym.empty:
                dfs_to_concat.append(df_synonym)

            df_augmented = pd.concat(dfs_to_concat, ignore_index=True)
            print(f"📈 Tổng dữ liệu sau augmentation: {len(df_augmented)} samples.")

            # 1.4. Save augmented dataset and update messages & labels
            augmented_file = original_file.replace('.csv', '_augmented.csv')
            df_augmented.to_csv(augmented_file, index=False)
            print(f"💾 Saved augmented data to: {augmented_file}")

            messages = df_augmented["Message"].tolist()
            labels = df_augmented["Category"].tolist()

        except Exception as e:
            print(f"⚠️ Augmentation failed: {e}")
            print("ℹ️ Continuing with original data...")
            df_augmented = original_df
    else:
        print("ℹ️ Data augmentation disabled")
        df_augmented = original_df

    # Return pipeline results
    results = {
        "original_count": len(original_df),
        "final_count": len(df_augmented),
        "augmented_count": len(df_augmented) - len(original_df),
        "messages": messages,
        "labels": labels,
        "original_file": original_file,
        "augmented_file": original_file.replace('.csv', '_augmented.csv') if use_augmentation else original_file,
        "source_type": source_type
    }

    print(f"\n📊 Pipeline Results:")
    print(f"   Original: {results['original_count']} samples")
    print(f"   Augmented: +{results['augmented_count']} samples")
    print(f"   Final: {results['final_count']} samples")

    return results


if __name__ == "__main__":
    TEST = 'kaggle'  # Default back to kaggle for consistency
    dataset_path = "data/2cls_spam_text_cls.csv"

    if os.path.exists("data/2cls_spam_text_cls.csv"):
        api_key = "a4910347ea0b1f86be877cd19899dd0bd3f855487a0b80eb611a64c0abf7a782"
        results_summary = {}
        classifier = SpamClassifier()

        # Test 1: Kaggle Dataset
        if TEST == 'kaggle':
            print("\n" + "="*60)
            print("TESTING WITH KAGGLE DATASET")
            print("="*60)
            messages, labels = classifier.load_dataset(source='kaggle')
            source_type = 'kaggle'

        # Test 2: Google Drive Dataset
        elif TEST == 'drive':
            print("\n" + "="*60)
            print("TESTING WITH GOOGLE DRIVE DATASET")
            print("="*60)
            messages, labels = classifier.load_dataset(source='gdrive', file_id='1N7rk-kfnDFIGMeX0ROVTjKh71gcgx-7R')
            source_type = 'gdrive'
        else:
            raise ValueError(f"Unknown TEST value: {TEST}")

        # Run enhanced pipeline with minimal testing mode
        print(f"🚀 Running enhanced pipeline with {len(messages)} messages...")
        pipeline_results = run_enhanced_pipeline(
            messages=messages,
            labels=labels,
            source_type=source_type,
            api_key=api_key,
            test_size=0.2,
            use_augmentation=True,
            augment_for_testing=True  # Use minimal augmentation for testing
        )

        print("\n🎉 Pipeline completed successfully!")
        print(f"📊 Results summary:")
        for key, value in pipeline_results.items():
            if key not in ['messages', 'labels']:  # Skip large data arrays
                print(f"   {key}: {value}")

        # Test the merged dataframe
        if pipeline_results['augmented_count'] > 0:
            print(f"\n✅ Successfully generated and merged augmented data!")
            print(f"   📁 Files created:")
            print(f"     - Original: {pipeline_results['original_file']}")
            print(f"     - Augmented: {pipeline_results['augmented_file']}")
        else:
            print(f"\n⚠️ No augmentation was performed")