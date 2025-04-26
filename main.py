import requests
from transformers import pipeline
from collections import Counter
import time
import webbrowser

# ========== Get Chrome tabs information ========== #
def get_open_tabs():
    try:
        tabs = requests.get('http://localhost:9222/json').json()
        return [(tab['title'], tab['url']) for tab in tabs if tab['type'] == 'page']
    except Exception as e:
        print("Make sure Chrome is started with remote-debugging.")
        return []

# ========== Load the model ========== #
# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define labels to classify into
labels = ["coding", "research", "other"]

def classify_tab(title, url):
    result = classifier(title, labels)
    return result['labels'][0]  # highest confidence label

def get_majority_topic(tabs):
    categories = [classify_tab(title, url) for title, url in tabs]
    print(f"Categories: {categories}")
    majority = Counter(categories).most_common(1)[0][0]
    return majority

# ========== Play music based on topic ========== #
def play_music_by_topic(majority):
    if majority == "coding":
        music_url = "https://www.youtube.com/watch?v=UfcAVejslrU"  # Rock playlist
    elif majority == "research":
        music_url = "https://www.youtube.com/watch?v=GSiqI-uwaN0&t=2706s"  # White noise
    else:
        music_url = "https://www.youtube.com/watch?v=igMqHb4r8vc"  # Taylor Swift playlist
    
    webbrowser.open(music_url)

# ========== Main function ========== #
def run_agent():
    tabs = get_open_tabs()
    print(f"Open tabs: {tabs}")
    if not tabs:
        print("No tabs found.")
        return

    topic = get_majority_topic(tabs)
    print(f"Detected topic: {topic}")
    play_music_by_topic(topic)


while True:
    run_agent()
    time.sleep(180)  # check every 3 minutes