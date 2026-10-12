import re
import html
import unicodedata
from datetime import datetime

from pymongo import UpdateOne

from config import REVIEWS_COLLECTION
from mongo_utils import get_database


# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 1000

MIN_CHARS_FOR_EMBEDDING = 20
MIN_WORDS_FOR_EMBEDDING = 4

MAX_CHARS_POSITIVE = 1200
MAX_CHARS_NEGATIVE = 1200
MAX_CHARS_EMBEDDING_TEXT = 2500

LANGUAGE_DETECTION_MIN_CHARS = 25

GENERIC_SHORT_TEXTS = {
    "ok",
    "okay",
    "good",
    "nice",
    "great",
    "bad",
    "nothing",
    "no",
    "none",
    "rien",
    "nulla",
    "tutto",
    "excellent",
    "very good",
    "perfect",
    "tốt",
    "rất tốt",
    "tuyệt",
    "tuyệt vời",
    "không",
    "không có",
    "ổn",
    "tệ",
}


# =========================================================
# OPTIONAL LANGUAGE DETECTION
# =========================================================

try:
    from langdetect import detect_langs
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False


# =========================================================
# TEXT CLEANING
# =========================================================

def safe_string(value):
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in ["", "nan", "none", "null"]:
        return None

    return text


def normalize_unicode(text):
    """
    Chuẩn hóa Unicode nhưng vẫn giữ dấu tiếng Việt, Pháp, Đức...
    """
    if text is None:
        return None

    return unicodedata.normalize("NFC", text)


def remove_urls(text):
    if text is None:
        return None

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    return text


def remove_emails(text):
    if text is None:
        return None

    return re.sub(r"\S+@\S+", " ", text)


def remove_control_characters(text):
    if text is None:
        return None

    return "".join(
        char for char in text
        if unicodedata.category(char)[0] != "C" or char in ["\n", "\t"]
    )


def normalize_whitespace(text):
    if text is None:
        return None

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_text(text):
    """
    Clean nhẹ, phù hợp dữ liệu đa ngôn ngữ.
    Không xóa dấu.
    Không xóa Unicode.
    Không xóa stopwords.
    Không stemming / lemmatization.
    """
    text = safe_string(text)

    if text is None:
        return None

    text = html.unescape(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_control_characters(text)
    text = normalize_whitespace(text)

    if text == "":
        return None

    return text


def truncate_text(text, max_chars):
    if text is None:
        return None

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].strip()


def build_cleaned_text(positive_clean, negative_clean):
    """
    Giữ tách biệt phần khen và phần chê bằng [SEP].
    """
    parts = []

    if positive_clean:
        parts.append(f"POSITIVE: {positive_clean}")

    if negative_clean:
        parts.append(f"NEGATIVE: {negative_clean}")

    if not parts:
        return None

    return " [SEP] ".join(parts)


def build_embedding_text(positive_clean, negative_clean):
    """
    Tạo text dùng cho embedding.
    Cắt riêng positive/negative để không mất phần negative ở cuối.
    """
    positive_part = truncate_text(positive_clean, MAX_CHARS_POSITIVE)
    negative_part = truncate_text(negative_clean, MAX_CHARS_NEGATIVE)

    text = build_cleaned_text(positive_part, negative_part)

    if text is None:
        return None

    return truncate_text(text, MAX_CHARS_EMBEDDING_TEXT)


def count_words(text):
    if text is None:
        return 0

    words = str(text).strip().split()

    return len(words)


def is_generic_short_text(text):
    if text is None:
        return True

    normalized = str(text).strip().lower()
    normalized = re.sub(r"[^\wÀ-ỹ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized in GENERIC_SHORT_TEXTS


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language(text):
    """
    Detect language nhẹ.
    Review quá ngắn thì để unknown vì dễ đoán sai.
    """
    if text is None:
        return "unknown", 0.0

    text = str(text).strip()

    if len(text) < LANGUAGE_DETECTION_MIN_CHARS:
        return "unknown", 0.0

    if not LANGDETECT_AVAILABLE:
        return "unknown", 0.0

    try:
        results = detect_langs(text)

        if not results:
            return "unknown", 0.0

        best = results[0]

        return best.lang, float(best.prob)

    except Exception:
        return "unknown", 0.0


# =========================================================
# SENTIMENT FROM RATING
# =========================================================

def sentiment_from_rating(rating):
    """
    Dùng rating làm sentiment chính vì dữ liệu đa ngôn ngữ.
    """
    try:
        score = float(rating)
    except Exception:
        return "unknown", None

    if score >= 8:
        return "positive", 1.0

    if score >= 6:
        return "neutral", 0.0

    return "negative", -1.0


# =========================================================
# VALIDATION FLAGS
# =========================================================

def get_text_flags(embedding_text):
    char_count = len(embedding_text) if embedding_text else 0
    word_count = count_words(embedding_text)

    is_short_review = (
        char_count < MIN_CHARS_FOR_EMBEDDING
        or word_count < MIN_WORDS_FOR_EMBEDDING
        or is_generic_short_text(embedding_text)
    )

    is_valid_for_embedding = (
        embedding_text is not None
        and not is_short_review
    )

    is_valid_for_aspect = is_valid_for_embedding

    return {
        "char_count": char_count,
        "word_count": word_count,
        "is_short_review": is_short_review,
        "is_valid_for_embedding": is_valid_for_embedding,
        "is_valid_for_aspect": is_valid_for_aspect,
    }


# =========================================================
# MAIN PREPROCESS
# =========================================================

def preprocess_reviews():
    db = get_database()
    reviews_col = db[REVIEWS_COLLECTION]

    query = {
        "review_text": {
            "$nin": [None, ""]
        }
    }

    total_reviews = reviews_col.count_documents(query)

    print("===== PREPROCESS REVIEWS =====")
    print(f"Tổng số review cần xử lý: {total_reviews}")
    print(f"Langdetect available: {LANGDETECT_AVAILABLE}")

    cursor = reviews_col.find(
        query,
        {
            "_id": 0,
            "review_id": 1,
            "positive_review": 1,
            "negative_review": 1,
            "review_text": 1,
            "rating": 1,
        }
    )

    operations = []

    processed_count = 0
    valid_embedding_count = 0
    short_review_count = 0
    unknown_language_count = 0

    for doc in cursor:
        review_id = doc["review_id"]

        positive_review = doc.get("positive_review")
        negative_review = doc.get("negative_review")
        review_text = doc.get("review_text")
        rating = doc.get("rating")

        positive_clean = clean_text(positive_review)
        negative_clean = clean_text(negative_review)

        # Trường hợp review_text có nhưng positive/negative đều rỗng
        # thì vẫn clean review_text để tránh mất dữ liệu.
        if positive_clean is None and negative_clean is None:
            cleaned_text = clean_text(review_text)
            embedding_text = truncate_text(cleaned_text, MAX_CHARS_EMBEDDING_TEXT)
        else:
            cleaned_text = build_cleaned_text(positive_clean, negative_clean)
            embedding_text = build_embedding_text(positive_clean, negative_clean)

        flags = get_text_flags(embedding_text)

        language, language_confidence = detect_language(embedding_text)

        sentiment_label, sentiment_score = sentiment_from_rating(rating)

        if flags["is_valid_for_embedding"]:
            valid_embedding_count += 1

        if flags["is_short_review"]:
            short_review_count += 1

        if language == "unknown":
            unknown_language_count += 1

        update_doc = {
            "positive_clean": positive_clean,
            "negative_clean": negative_clean,
            "cleaned_text": cleaned_text,
            "embedding_text": embedding_text,

            "char_count": flags["char_count"],
            "word_count": flags["word_count"],
            "is_short_review": flags["is_short_review"],
            "is_valid_for_embedding": flags["is_valid_for_embedding"],
            "is_valid_for_aspect": flags["is_valid_for_aspect"],

            "language": language,
            "language_confidence": language_confidence,

            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,

            "processed_status": "preprocessed",
            "preprocessed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        operations.append(
            UpdateOne(
                {"review_id": review_id},
                {"$set": update_doc}
            )
        )

        processed_count += 1

        if len(operations) >= BATCH_SIZE:
            reviews_col.bulk_write(operations, ordered=False)
            operations = []

            print(
                f"Đã xử lý {processed_count}/{total_reviews} review | "
                f"valid_embedding={valid_embedding_count} | "
                f"short={short_review_count} | "
                f"unknown_lang={unknown_language_count}"
            )

    if operations:
        reviews_col.bulk_write(operations, ordered=False)

    print("\n===== PREPROCESS FINISHED =====")
    print(f"Tổng số review đã xử lý: {processed_count}")
    print(f"Số review hợp lệ để embedding: {valid_embedding_count}")
    print(f"Số review ngắn / ít thông tin: {short_review_count}")
    print(f"Số review language unknown: {unknown_language_count}")


if __name__ == "__main__":
    preprocess_reviews()