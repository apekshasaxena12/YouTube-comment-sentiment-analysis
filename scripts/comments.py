import re
import csv
import os
import googleapiclient.discovery
import time

os.environ["YOUTUBE_API_KEY"] = "YOUR API KEY"

youtube = googleapiclient.discovery.build(
    "youtube",
    "v3",
    developerKey=os.getenv("YOUTUBE_API_KEY")
)


# TEXT PREPROCESSING FUNCTION
def preprocess_text(text):
    text = text.lower()                
    text = re.sub(r"\s+", " ", text)  
    return text.strip()


# FETCH COMMENTS
def fetch_comments(video_url, max_comments):
    if "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[-1].split("?")[0]
    elif "youtube.com/watch?v=" in video_url:
        video_id = video_url.split("v=")[-1].split("&")[0]
    else:
        raise ValueError("Invalid YouTube URL")

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText"
        )

        try:
            response = request.execute()
        except Exception as e:
            print("API request failed. Retrying...")
            time.sleep(5)
            continue


        for item in response.get("items", []):
            raw_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            cleaned_comment = preprocess_text(raw_comment)
            if cleaned_comment:
                comments.append(cleaned_comment)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments[:max_comments]


# SAVE COMMENTS TO CSV

def save_to_csv(comments, filename="youtube_comments.csv"):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["comment"])
        for comment in comments:
            writer.writerow([comment])

    print(f"\nSaved {len(comments)} comments to {filename}")


# MAIN FUNCTION
def main():
    video_url = input("Enter YouTube video URL: ").strip()
    n_comments = int(input("Enter number of comments to fetch: ").strip())

    print("\nFetching comments...")
    comments = fetch_comments(video_url, n_comments)

    print(f"Fetched {len(comments)} comments.")
    save_to_csv(comments)

if __name__ == "__main__":
    main()
