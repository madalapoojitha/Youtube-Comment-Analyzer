import gradio as gr
import pandas as pd
import re
import tempfile
from transformers import pipeline
from googleapiclient.discovery import build
import plotly.express as px

# Load pipelines
sentiment_pipeline = pipeline("sentiment-analysis")
toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"

def extract_video_id(url):
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
        r"youtube\.com\/shorts\/([^&\n?#]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_comments(video_url, max_results=10):
    video_id = extract_video_id(video_url)
    if not video_id:
        return pd.DataFrame({"error": ["Invalid YouTube URL"]})

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    comments = []
    try:
        response = request.execute()
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        return pd.DataFrame({"Comment": comments})
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def analyze_video(video_url, max_comments=10, sentiment_filter="All", toxicity_filter="All"):
    df = fetch_comments(video_url, max_comments)
    if "error" in df.columns:
        return df.to_string(index=False), None, None

    results = []
    for comment in df["Comment"]:
        sentiment_result = sentiment_pipeline(comment[:512])[0]
        toxic_results = toxic_classifier(comment[:512])
        toxic_labels = toxic_results[0]
        top_label = max(toxic_labels, key=lambda x: x['score'])

        sentiment = sentiment_result["label"]
        sentiment_score = round(sentiment_result["score"], 3)

        toxic_label = top_label["label"]
        toxic_score = round(top_label["score"], 3)
        toxic_tag = toxic_label if toxic_score > 0.5 else "Not Toxic"

        results.append({
            "Comment": comment,
            "Sentiment": sentiment,
            "Sentiment Score": sentiment_score,
            "Toxicity": toxic_tag,
            "Toxicity Score": toxic_score
        })

    result_df = pd.DataFrame(results)

    if sentiment_filter != "All":
        result_df = result_df[result_df["Sentiment"] == sentiment_filter]

    if toxicity_filter != "All":
        result_df = result_df[result_df["Toxicity"] == toxicity_filter]

    fig = px.histogram(result_df, x="Sentiment", title="Sentiment Distribution", color="Sentiment")
    fig.update_layout(bargap=0.2)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as f:
        result_df.to_csv(f.name, index=False)
        csv_file_path = f.name

    return result_df, fig, csv_file_path

# Interface
with gr.Blocks(title="Sentiment Dashboard") as demo:
    gr.HTML("""
    <style>
        body {
            background-color: #f4f4f9;
            font-family: 'Segoe UI', sans-serif;
        }
        .header {
            background: linear-gradient(to right, #3b82f6, #1e3a8a);
            color: white;
            padding: 2rem;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
        }
    </style>
    """)

    gr.HTML("""
    <div class="header">
        <h1>📊 YouTube Comment Analyzer</h1>
        <p>Analyze sentiment and toxicity from YouTube comments</p>
    </div>
    """)

    with gr.Row():
        video_url = gr.Textbox(label="YouTube Video URL", placeholder="Paste a YouTube video link")
        max_comments = gr.Slider(1, 100, value=10, step=1, label="Number of Comments")

    with gr.Row():
        sentiment_filter = gr.Dropdown(choices=["All", "POSITIVE", "NEGATIVE"], value="All", label="Filter by Sentiment")
        toxicity_filter = gr.Dropdown(choices=["All", "toxicity", "severe_toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit", "Not Toxic"], value="All", label="Filter by Toxicity")

    analyze_btn = gr.Button("🚀 Analyze Comments")

    with gr.Tabs():
        with gr.TabItem("📋 Analysis Table"):
            output_df = gr.Dataframe(label="Sentiment & Toxicity Table", interactive=False)

        with gr.TabItem("📈 Sentiment Chart"):
            output_plot = gr.Plot(label="Sentiment Distribution")

        with gr.TabItem("⬇️ Download CSV"):
            download_btn = gr.File(label="Download CSV")

    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_url, max_comments, sentiment_filter, toxicity_filter],
        outputs=[output_df, output_plot, download_btn]
    )

demo.launch()
