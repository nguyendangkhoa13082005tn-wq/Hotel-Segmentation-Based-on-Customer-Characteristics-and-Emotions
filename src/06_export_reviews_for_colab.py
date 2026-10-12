import pandas as pd

from config import REVIEWS_COLLECTION, TABLE_DIR
from mongo_utils import get_database


def export_reviews_for_colab():
    db = get_database()
    reviews_col = db[REVIEWS_COLLECTION]

    query = {
        "is_valid_for_embedding": True,
        "embedding_text": {
            "$nin": [None, ""]
        }
    }

    projection = {
        "_id": 0,
        "review_id": 1,
        "hotel_id": 1,
        "hotel_name": 1,
        "province_id": 1,
        "province_name": 1,
        "rating": 1,
        "sentiment_label": 1,
        "language": 1,
        "embedding_text": 1,

        # Cần cho Colab gán aspect bằng E5-Large
        "positive_clean": 1,
        "negative_clean": 1,
    }

    data = list(reviews_col.find(query, projection))
    df = pd.DataFrame(data)

    # Đảm bảo 2 cột này luôn tồn tại trong CSV
    for col in ["positive_clean", "negative_clean"]:
        if col not in df.columns:
            df[col] = ""

    # Đảm bảo các cột quan trọng luôn có mặt
    output_cols = [
        "review_id",
        "hotel_id",
        "hotel_name",
        "province_id",
        "province_name",
        "rating",
        "sentiment_label",
        "language",
        "embedding_text",
        "positive_clean",
        "negative_clean",
    ]

    for col in output_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[output_cols]

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # Đặt tên mới để tránh ghi đè file cũ đang mở bằng WPS/Excel
    output_path = TABLE_DIR / "reviews_for_colab_embedding_e5_aspect.csv"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("===== EXPORT REVIEWS FOR COLAB =====")
    print(f"Số review export: {len(df)}")
    print(f"Saved to: {output_path}")

    print("\n===== EXPORTED COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== CHECK MISSING =====")
    print(df[["embedding_text", "positive_clean", "negative_clean"]].isna().sum())

    print("\n===== SAMPLE =====")
    print(
        df[
            [
                "review_id",
                "language",
                "rating",
                "positive_clean",
                "negative_clean",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    export_reviews_for_colab()