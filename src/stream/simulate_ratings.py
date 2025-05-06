#!/usr/bin/env python
# @Auther liukun
# @Time 2025/04/13

import os, json, time, random, glob
from dotenv import load_dotenv

load_dotenv()
stream_path = f"{ROOT_DIR}/data/streaming_input"

MAX_KEEP = 10
DELAY = 3

os.makedirs(stream_path, exist_ok=True)


def create_user_ratings(i):
    rating = {
        "userId": random.randint(1, 500),
        "movieId": random.randint(1, 10000),
        "rating": round(random.uniform(1.0, 5.0), 1),
        "timestamp": int(time.time())
    }
    filename = f"{str(stream_path)}/rating_{int(time.time())}.json"
    # filename = f"{str(stream_path)}/rating_{i % 10}.json"
    with open(filename, "w") as f:
        json.dump(rating, f)
    i += 1
    print(f"write score：{rating}")


def clean_old_file():
    files = sorted(glob.glob(f"{str(stream_path)}/*.json"))
    if len(files) > MAX_KEEP:
        for old_file in files[:-MAX_KEEP]:
            os.remove(old_file)


def simulate_ratings():
    i = 0
    while True:
        create_user_ratings(i)
        clean_old_file()œ
        time.sleep(DELAY)


def refresh_old_files():
    while True:
        files = sorted(glob.glob(f"{str(stream_path)}/*.json"))
        if not files:
            print("⚠️ no JSON, waiting...")
            time.sleep(DELAY)
            continue

        for f in files:
            filename = os.path.basename(f)
            temp_file = f + ".tmp"
            os.rename(f, temp_file)
            print(f"[refresh] {filename} → {filename}.tmp")
            time.sleep(0.3)
            os.rename(temp_file, f)
            print(f"[refresh] {filename}.tmp → {filename}")
            time.sleep(DELAY)

        print(f"next round ...")
        time.sleep(DELAY)


if __name__ == "__main__":
    simulate_ratings()
    refresh_old_files()