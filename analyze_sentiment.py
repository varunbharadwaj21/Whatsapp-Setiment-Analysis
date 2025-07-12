import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt

def parse_chat(file_path):
    messages = []
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        match = re.match(r'^(\d{2}/\d{2}/\d{4}),\s(\d{1,2}:\d{2}\s[APMapm]{2})\s-\s(.+?):\s(.+)', line)
        if match:
            date, time, sender, message = match.groups()
            messages.append([date, time, sender, message])
    return pd.DataFrame(messages, columns=["Date", "Time", "Sender", "Message"])

def analyze_sentiment(df):
    df["Polarity"] = df["Message"].apply(lambda text: TextBlob(text).sentiment.polarity)
    df["Sentiment"] = df["Polarity"].apply(lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral")
    return df

def plot_sentiments(df):
    sentiment_counts = df["Sentiment"].value_counts()
    sentiment_counts.plot(kind="bar", color=["green", "red", "gray"])
    plt.title("WhatsApp Chat Sentiment Analysis")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Messages")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = parse_chat("chat.txt")
    df = analyze_sentiment(df)
    print(df.head())
    plot_sentiments(df)
