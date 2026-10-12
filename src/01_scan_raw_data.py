import pandas as pd

from config import RAW_DATA_DIR, TABLE_DIR


def scan_raw_data():
    rows = []

    if not RAW_DATA_DIR.exists():
        print(f"Không tìm thấy thư mục: {RAW_DATA_DIR}")
        return

    for province_dir in RAW_DATA_DIR.iterdir():
        if not province_dir.is_dir():
            continue

        province_id = province_dir.name
        province_name = province_id.replace("-", " ").title()

        data_files = [
            file for file in province_dir.rglob("*")
            if file.is_file() and file.suffix.lower() in [".csv", ".json", ".xlsx"]
        ]

        rows.append({
            "province_id": province_id,
            "province_name": province_name,
            "num_files": len(data_files),
            "source_folder": str(province_dir)
        })

    df = pd.DataFrame(rows)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLE_DIR / "raw_data_scan_summary.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("===== RAW DATA SCAN SUMMARY =====")
    print(df)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    scan_raw_data()