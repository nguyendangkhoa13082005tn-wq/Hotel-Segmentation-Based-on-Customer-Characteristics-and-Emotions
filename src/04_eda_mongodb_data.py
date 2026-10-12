import re
from collections import Counter

import pandas as pd

from config import (
    REVIEWS_COLLECTION,
    HOTELS_COLLECTION,
    TABLE_DIR
)
from mongo_utils import get_database


def clean_for_length(text):
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_hotels_df(db):
    hotels_col = db[HOTELS_COLLECTION]

    cursor = hotels_col.find(
        {},
        {
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
    )

    return pd.DataFrame(list(cursor))


def load_reviews_df(db):
    reviews_col = db[REVIEWS_COLLECTION]

    cursor = reviews_col.find(
        {},
        {
            "_id": 0,
            "review_id": 1,
            "hotel_id": 1,
            "hotel_name": 1,
            "province_id": 1,
            "province_name": 1,
            "customer_country": 1,
            "review_date": 1,
            "rating": 1,
            "review_rating_group": 1,
            "room_type": 1,
            "stay_duration": 1,
            "group_type": 1,
            "positive_review": 1,
            "negative_review": 1,
            "review_text": 1,
            "language": 1,
            "processed_status": 1,
            "is_embedding_sample": 1,
            "source_file": 1,
        }
    )

    return pd.DataFrame(list(cursor))


def export_basic_summary(hotels_df, reviews_df):
    summary = {
        "num_hotels": len(hotels_df),
        "num_reviews": len(reviews_df),
        "num_provinces_hotels": hotels_df["province_id"].nunique() if not hotels_df.empty else 0,
        "num_provinces_reviews": reviews_df["province_id"].nunique() if not reviews_df.empty else 0,
        "num_customer_countries": reviews_df["customer_country"].nunique(dropna=True) if not reviews_df.empty else 0,
        "num_reviews_with_text": int(reviews_df["review_text"].notna().sum()) if "review_text" in reviews_df.columns else 0,
        "num_reviews_without_text": int(reviews_df["review_text"].isna().sum()) if "review_text" in reviews_df.columns else 0,
        "avg_reviews_per_hotel": round(len(reviews_df) / hotels_df["hotel_id"].nunique(), 2) if not hotels_df.empty else 0,
    }

    summary_df = pd.DataFrame([summary])
    output_path = TABLE_DIR / "eda_01_basic_summary.csv"
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== BASIC SUMMARY =====")
    print(summary_df)
    print(f"Saved: {output_path}")


def export_province_summary(hotels_df, reviews_df):
    hotel_summary = hotels_df.groupby("province_id").agg(
        num_hotels=("hotel_id", "nunique"),
        avg_hotel_rating=("hotel_rating", "mean"),
        high_hotels=("hotel_rating_group", lambda x: (x == "high").sum()),
        mid_hotels=("hotel_rating_group", lambda x: (x == "mid").sum()),
        low_hotels=("hotel_rating_group", lambda x: (x == "low").sum()),
    ).reset_index()

    review_summary = reviews_df.groupby("province_id").agg(
        num_reviews=("review_id", "count"),
        num_review_hotels=("hotel_id", "nunique"),
        avg_review_score=("rating", "mean"),
        positive_reviews=("review_rating_group", lambda x: (x == "positive").sum()),
        neutral_reviews=("review_rating_group", lambda x: (x == "neutral").sum()),
        negative_reviews=("review_rating_group", lambda x: (x == "negative").sum()),
        num_customer_countries=("customer_country", "nunique"),
    ).reset_index()

    province_summary = hotel_summary.merge(
        review_summary,
        on="province_id",
        how="outer"
    )

    province_summary["avg_hotel_rating"] = province_summary["avg_hotel_rating"].round(3)
    province_summary["avg_review_score"] = province_summary["avg_review_score"].round(3)

    province_summary["positive_ratio"] = (
        province_summary["positive_reviews"] / province_summary["num_reviews"]
    ).round(4)

    province_summary["neutral_ratio"] = (
        province_summary["neutral_reviews"] / province_summary["num_reviews"]
    ).round(4)

    province_summary["negative_ratio"] = (
        province_summary["negative_reviews"] / province_summary["num_reviews"]
    ).round(4)

    output_path = TABLE_DIR / "eda_02_province_summary.csv"
    province_summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== PROVINCE SUMMARY =====")
    print(province_summary.head(20))
    print(f"Saved: {output_path}")


def export_hotel_summary(hotels_df, reviews_df):
    review_by_hotel = reviews_df.groupby(["hotel_id", "province_id"]).agg(
        num_reviews_sample=("review_id", "count"),
        avg_review_score=("rating", "mean"),
        positive_reviews=("review_rating_group", lambda x: (x == "positive").sum()),
        neutral_reviews=("review_rating_group", lambda x: (x == "neutral").sum()),
        negative_reviews=("review_rating_group", lambda x: (x == "negative").sum()),
        num_customer_countries=("customer_country", "nunique"),
    ).reset_index()

    hotel_cols = [
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
    ]

    hotel_summary = hotels_df[hotel_cols].merge(
        review_by_hotel,
        on=["hotel_id", "province_id"],
        how="left"
    )

    hotel_summary["avg_review_score"] = hotel_summary["avg_review_score"].round(3)

    hotel_summary["positive_ratio"] = (
        hotel_summary["positive_reviews"] / hotel_summary["num_reviews_sample"]
    ).round(4)

    hotel_summary["neutral_ratio"] = (
        hotel_summary["neutral_reviews"] / hotel_summary["num_reviews_sample"]
    ).round(4)

    hotel_summary["negative_ratio"] = (
        hotel_summary["negative_reviews"] / hotel_summary["num_reviews_sample"]
    ).round(4)

    hotel_summary = hotel_summary.sort_values(
        ["province_id", "num_reviews_sample"],
        ascending=[True, False]
    )

    output_path = TABLE_DIR / "eda_03_hotel_summary.csv"
    hotel_summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== HOTEL SUMMARY =====")
    print(hotel_summary.head(20))
    print(f"Saved: {output_path}")


def export_review_length_summary(reviews_df):
    df = reviews_df.copy()

    df["review_text_clean_for_length"] = df["review_text"].apply(clean_for_length)
    df["char_count"] = df["review_text_clean_for_length"].str.len()
    df["word_count"] = df["review_text_clean_for_length"].str.split().str.len()

    length_summary = {
        "num_reviews": len(df),
        "min_char_count": int(df["char_count"].min()),
        "max_char_count": int(df["char_count"].max()),
        "avg_char_count": round(df["char_count"].mean(), 2),
        "median_char_count": round(df["char_count"].median(), 2),
        "min_word_count": int(df["word_count"].min()),
        "max_word_count": int(df["word_count"].max()),
        "avg_word_count": round(df["word_count"].mean(), 2),
        "median_word_count": round(df["word_count"].median(), 2),
        "reviews_less_5_words": int((df["word_count"] < 5).sum()),
        "reviews_less_10_words": int((df["word_count"] < 10).sum()),
        "reviews_more_200_words": int((df["word_count"] > 200).sum()),
    }

    summary_df = pd.DataFrame([length_summary])
    output_path = TABLE_DIR / "eda_04_review_length_summary.csv"
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    review_length_detail = df[
        [
            "review_id",
            "hotel_id",
            "hotel_name",
            "province_id",
            "rating",
            "review_rating_group",
            "char_count",
            "word_count",
            "review_text",
        ]
    ].sort_values("word_count", ascending=False)

    detail_path = TABLE_DIR / "eda_04_review_length_detail.csv"
    review_length_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("\n===== REVIEW LENGTH SUMMARY =====")
    print(summary_df)
    print(f"Saved: {output_path}")
    print(f"Saved: {detail_path}")


def export_country_summary(reviews_df):
    country_summary = reviews_df.groupby("customer_country").agg(
        num_reviews=("review_id", "count"),
        num_hotels=("hotel_id", "nunique"),
        num_provinces=("province_id", "nunique"),
        avg_score=("rating", "mean"),
        positive_reviews=("review_rating_group", lambda x: (x == "positive").sum()),
        neutral_reviews=("review_rating_group", lambda x: (x == "neutral").sum()),
        negative_reviews=("review_rating_group", lambda x: (x == "negative").sum()),
    ).reset_index()

    country_summary["avg_score"] = country_summary["avg_score"].round(3)

    country_summary = country_summary.sort_values(
        "num_reviews",
        ascending=False
    )

    output_path = TABLE_DIR / "eda_05_customer_country_summary.csv"
    country_summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== CUSTOMER COUNTRY SUMMARY =====")
    print(country_summary.head(30))
    print(f"Saved: {output_path}")


def export_room_group_summary(reviews_df):
    group_summary = reviews_df.groupby("group_type").agg(
        num_reviews=("review_id", "count"),
        avg_score=("rating", "mean"),
        positive_reviews=("review_rating_group", lambda x: (x == "positive").sum()),
        neutral_reviews=("review_rating_group", lambda x: (x == "neutral").sum()),
        negative_reviews=("review_rating_group", lambda x: (x == "negative").sum()),
    ).reset_index()

    group_summary["avg_score"] = group_summary["avg_score"].round(3)
    group_summary = group_summary.sort_values("num_reviews", ascending=False)

    group_path = TABLE_DIR / "eda_06_group_type_summary.csv"
    group_summary.to_csv(group_path, index=False, encoding="utf-8-sig")

    room_summary = reviews_df.groupby("room_type").agg(
        num_reviews=("review_id", "count"),
        avg_score=("rating", "mean"),
        positive_reviews=("review_rating_group", lambda x: (x == "positive").sum()),
        neutral_reviews=("review_rating_group", lambda x: (x == "neutral").sum()),
        negative_reviews=("review_rating_group", lambda x: (x == "negative").sum()),
    ).reset_index()

    room_summary["avg_score"] = room_summary["avg_score"].round(3)
    room_summary = room_summary.sort_values("num_reviews", ascending=False)

    room_path = TABLE_DIR / "eda_07_room_type_summary.csv"
    room_summary.to_csv(room_path, index=False, encoding="utf-8-sig")

    print("\n===== GROUP TYPE SUMMARY =====")
    print(group_summary.head(20))
    print(f"Saved: {group_path}")

    print("\n===== ROOM TYPE SUMMARY =====")
    print(room_summary.head(20))
    print(f"Saved: {room_path}")


def export_text_keyword_preview(reviews_df):
    """
    Thống kê từ xuất hiện nhiều.
    Chỉ để tham khảo nhanh, không phải xử lý NLP chính.
    Không xóa stopwords mạnh vì dữ liệu đa ngôn ngữ.
    """
    all_text = " ".join(
        reviews_df["review_text"]
        .fillna("")
        .astype(str)
        .tolist()
    ).lower()

    words = re.findall(r"[a-zA-ZÀ-ỹ一-龥ぁ-んァ-ン가-힣]+", all_text)

    # Bỏ một số từ quá phổ biến, rất cơ bản
    stopwords = {
        "the", "and", "to", "a", "of", "in", "was", "is", "it", "for", "with",
        "i", "we", "you", "that", "this", "very", "but", "not",
        "và", "là", "có", "rất", "không", "cho", "một", "ở", "được", "này"
    }

    words = [w for w in words if len(w) >= 3 and w not in stopwords]

    counter = Counter(words)

    keyword_df = pd.DataFrame(
        counter.most_common(300),
        columns=["word", "count"]
    )

    output_path = TABLE_DIR / "eda_08_top_keywords_preview.csv"
    keyword_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== TOP KEYWORDS PREVIEW =====")
    print(keyword_df.head(50))
    print(f"Saved: {output_path}")


def export_sample_reviews_for_manual_check(reviews_df):
    cols = [
        "review_id",
        "hotel_name",
        "province_id",
        "customer_country",
        "rating",
        "review_rating_group",
        "positive_review",
        "negative_review",
        "review_text",
    ]

    # Lấy mẫu mỗi nhóm điểm để bạn đọc thủ công
    samples = []

    for group in ["positive", "neutral", "negative"]:
        group_df = reviews_df[reviews_df["review_rating_group"] == group]
        samples.append(group_df[cols].head(100))

    sample_df = pd.concat(samples, ignore_index=True)

    output_path = TABLE_DIR / "eda_09_sample_reviews_for_manual_check.csv"
    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== SAMPLE REVIEWS FOR MANUAL CHECK =====")
    print(sample_df.head(20))
    print(f"Saved: {output_path}")


def export_missing_values(hotels_df, reviews_df):
    hotel_missing = pd.DataFrame({
        "column": hotels_df.columns,
        "missing_count": [hotels_df[col].isna().sum() for col in hotels_df.columns],
        "missing_ratio": [round(hotels_df[col].isna().mean(), 4) for col in hotels_df.columns],
        "dataset": "hotels"
    })

    review_missing = pd.DataFrame({
        "column": reviews_df.columns,
        "missing_count": [reviews_df[col].isna().sum() for col in reviews_df.columns],
        "missing_ratio": [round(reviews_df[col].isna().mean(), 4) for col in reviews_df.columns],
        "dataset": "reviews"
    })

    missing_df = pd.concat([hotel_missing, review_missing], ignore_index=True)
    output_path = TABLE_DIR / "eda_10_missing_values.csv"
    missing_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n===== MISSING VALUES =====")
    print(missing_df.sort_values("missing_ratio", ascending=False).head(30))
    print(f"Saved: {output_path}")


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    db = get_database()

    print("===== EDA MONGODB DATA =====")

    hotels_df = load_hotels_df(db)
    reviews_df = load_reviews_df(db)

    if hotels_df.empty:
        print("Không có dữ liệu hotels.")
        return

    if reviews_df.empty:
        print("Không có dữ liệu reviews.")
        return

    export_basic_summary(hotels_df, reviews_df)
    export_province_summary(hotels_df, reviews_df)
    export_hotel_summary(hotels_df, reviews_df)
    export_review_length_summary(reviews_df)
    export_country_summary(reviews_df)
    export_room_group_summary(reviews_df)
    export_text_keyword_preview(reviews_df)
    export_sample_reviews_for_manual_check(reviews_df)
    export_missing_values(hotels_df, reviews_df)

    print("\n===== EDA FINISHED =====")
    print(f"Các file EDA đã được lưu tại: {TABLE_DIR}")


if __name__ == "__main__":
    main()