import math
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from pymongo import UpdateOne

from config import (
    REVIEWS_COLLECTION,
    HOTELS_COLLECTION,
    HOTEL_FEATURES_COLLECTION,
    TABLE_DIR
)
from mongo_utils import get_database


# =========================================================
# CONFIG
# =========================================================

N_REVIEW_TOPICS = 12
BATCH_SIZE = 500

ASPECTS = [
    "room",
    "cleanliness",
    "service",
    "location",
    "price",
    "food",
    "facilities",
    "wifi",
    "noise",
    "checkin"
]


# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = " ".join(text.split())

    return text


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def to_python_value(value):
    """
    Chuyển numpy/pandas type thành Python type để lưu MongoDB.
    """
    if isinstance(value, (list, dict)):
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def clean_mongo_dict(data):
    cleaned = {}

    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = clean_mongo_dict(value)
        elif isinstance(value, list):
            cleaned[key] = [to_python_value(v) for v in value]
        else:
            cleaned[key] = to_python_value(value)

    return cleaned


def entropy_from_counts(counts):
    total = sum(counts)

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counts:
        if count <= 0:
            continue

        p = count / total
        entropy -= p * math.log(p)

    return float(entropy)


# =========================================================
# LOAD DATA
# =========================================================

def load_reviews_df(db):
    reviews_col = db[REVIEWS_COLLECTION]

    projection = {
        "_id": 0,
        "review_id": 1,
        "hotel_id": 1,
        "hotel_name": 1,
        "province_id": 1,
        "province_name": 1,

        "rating": 1,
        "sentiment_label": 1,
        "sentiment_score": 1,
        "review_rating_group": 1,

        "char_count": 1,
        "word_count": 1,
        "is_short_review": 1,
        "is_valid_for_embedding": 1,

        "customer_country": 1,
        "group_type": 1,
        "room_type": 1,

        "review_topic": 1,

        # Aspect đã được gán từ Colab E5-Large
        "positive_aspects": 1,
        "negative_aspects": 1,
        "review_aspects": 1,
        "aspect_method": 1,
        "aspect_model": 1,
    }

    data = list(reviews_col.find({}, projection))

    return pd.DataFrame(data)


def load_hotels_df(db):
    hotels_col = db[HOTELS_COLLECTION]

    projection = {
        "_id": 0,
        "hotel_id": 1,
        "hotel_name": 1,
        "province_id": 1,
        "province_name": 1,

        "hotel_rating": 1,
        "hotel_rating_group": 1,
        "total_rating": 1,
        "num_reviews_in_file": 1,
        "num_reviews_with_text_in_file": 1,
        "num_reviews_imported_sample": 1,

        "evaluation_categories": 1,
        "source_file": 1,
    }

    data = list(hotels_col.find({}, projection))

    return pd.DataFrame(data)


# =========================================================
# FEATURE BUILDERS
# =========================================================

def build_rating_sentiment_features(reviews_df):
    rows = []

    for hotel_id, group in reviews_df.groupby("hotel_id"):
        ratings = pd.to_numeric(group["rating"], errors="coerce")
        sentiment_scores = pd.to_numeric(group["sentiment_score"], errors="coerce")

        num_reviews = len(group)

        positive_count = (group["sentiment_label"] == "positive").sum()
        neutral_count = (group["sentiment_label"] == "neutral").sum()
        negative_count = (group["sentiment_label"] == "negative").sum()

        rows.append({
            "hotel_id": hotel_id,
            "num_reviews": int(num_reviews),

            "avg_review_rating": float(ratings.mean()) if ratings.notna().any() else None,
            "std_review_rating": float(ratings.std(ddof=0)) if ratings.notna().any() else 0.0,
            "min_review_rating": float(ratings.min()) if ratings.notna().any() else None,
            "max_review_rating": float(ratings.max()) if ratings.notna().any() else None,

            "positive_review_count": int(positive_count),
            "neutral_review_count": int(neutral_count),
            "negative_review_count": int(negative_count),

            "positive_ratio": float(positive_count / num_reviews) if num_reviews else 0.0,
            "neutral_ratio": float(neutral_count / num_reviews) if num_reviews else 0.0,
            "negative_ratio": float(negative_count / num_reviews) if num_reviews else 0.0,

            "avg_sentiment_score": float(sentiment_scores.mean()) if sentiment_scores.notna().any() else 0.0,
        })

    return pd.DataFrame(rows)


def build_text_length_features(reviews_df):
    rows = []

    for hotel_id, group in reviews_df.groupby("hotel_id"):
        char_counts = pd.to_numeric(group["char_count"], errors="coerce").fillna(0)
        word_counts = pd.to_numeric(group["word_count"], errors="coerce").fillna(0)

        num_reviews = len(group)
        short_count = (group["is_short_review"] == True).sum()
        valid_embedding_count = (group["is_valid_for_embedding"] == True).sum()

        rows.append({
            "hotel_id": hotel_id,
            "avg_char_count": float(char_counts.mean()),
            "avg_word_count": float(word_counts.mean()),
            "max_word_count": int(word_counts.max()) if len(word_counts) else 0,
            "short_review_ratio": float(short_count / num_reviews) if num_reviews else 0.0,
            "valid_embedding_ratio": float(valid_embedding_count / num_reviews) if num_reviews else 0.0,
        })

    return pd.DataFrame(rows)


def build_country_features(reviews_df):
    rows = []

    for hotel_id, group in reviews_df.groupby("hotel_id"):
        countries = group["customer_country"].dropna().astype(str)
        num_reviews = len(group)

        if len(countries) == 0:
            rows.append({
                "hotel_id": hotel_id,
                "num_customer_countries": 0,
                "top_country_ratio": 0.0,
                "country_diversity": 0.0,
            })
            continue

        counter = Counter(countries)
        counts = list(counter.values())
        top_country_count = max(counts)

        rows.append({
            "hotel_id": hotel_id,
            "num_customer_countries": int(len(counter)),
            "top_country_ratio": float(top_country_count / num_reviews) if num_reviews else 0.0,
            "country_diversity": entropy_from_counts(counts),
        })

    return pd.DataFrame(rows)


def normalize_group_type(value):
    text = normalize_text(value)

    if not text:
        return "unknown"

    if "cặp" in text or "couple" in text:
        return "couple"

    if "gia đình" in text or "family" in text:
        return "family"

    if "khách lẻ" in text or "solo" in text or "alone" in text:
        return "solo"

    if "nhóm" in text or "group" in text or "friends" in text:
        return "group"

    return "other"


def build_group_type_features(reviews_df):
    df = reviews_df.copy()

    if "group_type" not in df.columns:
        df["group_type"] = None

    df["group_type_norm"] = df["group_type"].apply(normalize_group_type)

    group_types = ["couple", "family", "solo", "group", "other", "unknown"]

    rows = []

    for hotel_id, group in df.groupby("hotel_id"):
        num_reviews = len(group)

        row = {
            "hotel_id": hotel_id
        }

        for group_type in group_types:
            count = (group["group_type_norm"] == group_type).sum()
            row[f"{group_type}_ratio"] = float(count / num_reviews) if num_reviews else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_topic_features(reviews_df):
    rows = []

    for hotel_id, group in reviews_df.groupby("hotel_id"):
        num_reviews = len(group)

        row = {
            "hotel_id": hotel_id
        }

        for topic in range(N_REVIEW_TOPICS):
            count = (group["review_topic"] == topic).sum()
            row[f"topic_{topic}_ratio"] = float(count / num_reviews) if num_reviews else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_aspect_features(reviews_df):
    """
    Tổng hợp aspect đã được gán sẵn trong reviews.
    File này không tự gán aspect nữa.
    """
    rows = []

    for hotel_id, group in reviews_df.groupby("hotel_id"):
        num_reviews = len(group)

        row = {
            "hotel_id": hotel_id
        }

        positive_aspect_counts = defaultdict(int)
        negative_aspect_counts = defaultdict(int)
        any_aspect_counts = defaultdict(int)

        for _, review in group.iterrows():
            positive_aspects = review.get("positive_aspects", [])
            negative_aspects = review.get("negative_aspects", [])
            review_aspects = review.get("review_aspects", [])

            if not isinstance(positive_aspects, list):
                positive_aspects = []

            if not isinstance(negative_aspects, list):
                negative_aspects = []

            if not isinstance(review_aspects, list):
                review_aspects = []

            for aspect in positive_aspects:
                positive_aspect_counts[aspect] += 1

            for aspect in negative_aspects:
                negative_aspect_counts[aspect] += 1

            for aspect in review_aspects:
                any_aspect_counts[aspect] += 1

        for aspect in ASPECTS:
            row[f"{aspect}_positive_ratio"] = float(
                positive_aspect_counts[aspect] / num_reviews
            ) if num_reviews else 0.0

            row[f"{aspect}_negative_ratio"] = float(
                negative_aspect_counts[aspect] / num_reviews
            ) if num_reviews else 0.0

            row[f"{aspect}_mention_ratio"] = float(
                any_aspect_counts[aspect] / num_reviews
            ) if num_reviews else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def flatten_evaluation_categories(hotels_df):
    rows = []

    for _, row in hotels_df.iterrows():
        hotel_id = row["hotel_id"]
        categories = row.get("evaluation_categories", {})

        if not isinstance(categories, dict):
            categories = {}

        output = {
            "hotel_id": hotel_id,
            "service_staff": safe_float(categories.get("service_staff"), 0.0),
            "amenities": safe_float(categories.get("amenities"), 0.0),
            "cleanliness": safe_float(categories.get("cleanliness"), 0.0),
            "comfort": safe_float(categories.get("comfort"), 0.0),
            "value_for_money": safe_float(categories.get("value_for_money"), 0.0),
            "location": safe_float(categories.get("location"), 0.0),
            "free_wifi": safe_float(categories.get("free_wifi"), 0.0),
        }

        rows.append(output)

    return pd.DataFrame(rows)


# =========================================================
# BUILD HOTEL FEATURES
# =========================================================

def build_hotel_features_dataframe(hotels_df, reviews_df):
    base_cols = [
        "hotel_id",
        "hotel_name",
        "province_id",
        "province_name",
        "hotel_rating",
        "hotel_rating_group",
        "total_rating",
        "num_reviews_in_file",
        "num_reviews_with_text_in_file",
        "num_reviews_imported_sample",
        "source_file",
    ]

    for col in base_cols:
        if col not in hotels_df.columns:
            hotels_df[col] = None

    base_df = hotels_df[base_cols].copy()

    rating_features = build_rating_sentiment_features(reviews_df)
    text_features = build_text_length_features(reviews_df)
    country_features = build_country_features(reviews_df)
    group_features = build_group_type_features(reviews_df)
    topic_features = build_topic_features(reviews_df)
    aspect_features = build_aspect_features(reviews_df)
    eval_features = flatten_evaluation_categories(hotels_df)

    feature_df = base_df.merge(rating_features, on="hotel_id", how="left")
    feature_df = feature_df.merge(text_features, on="hotel_id", how="left")
    feature_df = feature_df.merge(country_features, on="hotel_id", how="left")
    feature_df = feature_df.merge(group_features, on="hotel_id", how="left")
    feature_df = feature_df.merge(topic_features, on="hotel_id", how="left")
    feature_df = feature_df.merge(aspect_features, on="hotel_id", how="left")
    feature_df = feature_df.merge(eval_features, on="hotel_id", how="left")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)

    feature_df["hotel_feature_status"] = "built_from_existing_e5_aspects"
    feature_df["feature_built_at"] = datetime.utcnow()
    feature_df["updated_at"] = datetime.utcnow()

    return feature_df


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_hotel_features_to_mongodb(db, feature_df):
    hotel_features_col = db[HOTEL_FEATURES_COLLECTION]

    operations = []

    for _, row in feature_df.iterrows():
        data = row.to_dict()
        data = clean_mongo_dict(data)

        hotel_id = data["hotel_id"]

        operations.append(
            UpdateOne(
                {"hotel_id": hotel_id},
                {"$set": data},
                upsert=True
            )
        )

        if len(operations) >= BATCH_SIZE:
            hotel_features_col.bulk_write(operations, ordered=False)
            operations = []

    if operations:
        hotel_features_col.bulk_write(operations, ordered=False)

    print(f"Đã lưu {len(feature_df)} hotel_features vào MongoDB.")


def export_hotel_features(feature_df):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLE_DIR / "hotel_features.csv"
    feature_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_cols_df = pd.DataFrame({
        "numeric_feature": numeric_cols
    })

    numeric_cols_path = TABLE_DIR / "hotel_numeric_features.csv"
    numeric_cols_df.to_csv(numeric_cols_path, index=False, encoding="utf-8-sig")

    print(f"Saved hotel features: {output_path}")
    print(f"Saved numeric feature list: {numeric_cols_path}")


def print_summary(feature_df):
    print("\n===== HOTEL FEATURES SUMMARY =====")
    print(f"Số khách sạn có feature: {len(feature_df)}")
    print(f"Số cột feature: {feature_df.shape[1]}")

    important_cols = [
        "hotel_id",
        "hotel_name",
        "province_id",
        "num_reviews",
        "avg_review_rating",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "avg_sentiment_score",
        "num_customer_countries",
        "country_diversity",
        "service_staff",
        "cleanliness",
        "location",
        "value_for_money",
        "room_positive_ratio",
        "room_negative_ratio",
        "service_positive_ratio",
        "service_negative_ratio",
        "cleanliness_positive_ratio",
        "cleanliness_negative_ratio",
    ]

    existing_cols = [col for col in important_cols if col in feature_df.columns]

    print("\nMột số cột chính:")
    print(feature_df[existing_cols].head(10))


# =========================================================
# MAIN
# =========================================================

def main():
    db = get_database()

    print("===== BUILD HOTEL FEATURES =====")
    print("===== SKIP LOCAL ASPECT ASSIGNMENT: USING E5 ASPECT FROM COLAB =====")

    print("\n===== LOAD DATA =====")
    hotels_df = load_hotels_df(db)
    reviews_df = load_reviews_df(db)

    print(f"Số hotels: {len(hotels_df)}")
    print(f"Số reviews: {len(reviews_df)}")

    if hotels_df.empty:
        print("Không có dữ liệu hotels.")
        return

    if reviews_df.empty:
        print("Không có dữ liệu reviews.")
        return

    feature_df = build_hotel_features_dataframe(hotels_df, reviews_df)

    save_hotel_features_to_mongodb(db, feature_df)
    export_hotel_features(feature_df)
    print_summary(feature_df)

    print("\n===== FILE 07 FINISHED =====")


if __name__ == "__main__":
    main()