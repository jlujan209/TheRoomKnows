from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib import colors
import matplotlib.pyplot as plt
from SpeechAnalysis import group_by_qa as gqa
import os
import json

def main():
    chat_output = "output-doc-pat.json"
    if not os.path.exists(chat_output):
        print("Chat output file does not exist. Generating new file.")
        response = gqa.query_openai("SpeechAnalysis/test.json", "output-doc-pat.json")
    else:
        print("Chat output file already exists. Loading from file.")
        with open(chat_output, "r") as f:
            response = json.load(f)

    sentiment_counts = gqa.perform_sentiment_analysis(response)
    sentiment_plot = generate_sentiment_plot(sentiment_counts, "sentiment_plot.png")

    generate_pdf_report(sentiment_plot, "reports/emotion_report_2025-02-20_14-49-51.txt", "report.pdf")

def generate_sentiment_plot(sentiment_counts: dict, output_file: str):
    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())

    plt.bar(range(len(sentiment_counts)), counts, tick_label=labels)
    plt.title("Sentiment Analysis of Doctor Visit")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")

    plt.savefig(output_file)
    plt.close() 

    return output_file

def generate_emotion_plot(emotion_data_path: str, output_file: str):
    emotion_data = {}
    with open(emotion_data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(",")
                count = parts[0].split(": ")
                emotion_data[count[0]] = int(count[1])
    
    labels = list(emotion_data.keys())
    counts = list(emotion_data.values())

    plt.bar(range(len(emotion_data)), counts, tick_label=labels)
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")

    plt.savefig(output_file)
    plt.close()

    return output_file

def generate_pdf_report(sentiment_plot: str, emotion_data: str, output_file: str = "report.pdf"):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    cur_y = height - 50

    # Add title
    c.setFont("Helvetica", 16)
    c.drawString(100, cur_y, "Doctor Visit Analysis Report")
    cur_y -= 20
    c.drawString(100, cur_y, "SUBJECTIVE")
    cur_y -= 20
    c.drawString(100, cur_y, "Emotion Analysis Output")
    cur_y -= 320
    emotion_plot = generate_emotion_plot(emotion_data, "emotion_plot.png")
    c.drawImage(emotion_plot, 100, cur_y, width=400, height=300)

    cur_y -= 20
    c.drawString(100, cur_y, "OBJECTIVE")
    cur_y -= 20
    # Add sentiment plot
    c.drawString(100, cur_y, "Sentiment Analysis Output")
    cur_y -= 320
    c.drawImage(sentiment_plot, 100, cur_y, width=400, height=300)
    cur_y -= 50
    c.drawString(100, cur_y, "Chief Complaint Counts")
    # Create a table for the counts
    data = [["Symptom", "Count"]]
    symptoms = {
        "symptoms": [
            {"symptom": "chest pain", "count": 6},
            {"symptom": "shortness of breath", "count": 4},
            {"symptom": "sweating", "count": 3},
            {"symptom": "palpitations", "count": 1},
        ]
    }
    for symptom in symptoms["symptoms"]:
        data.append([symptom["symptom"], symptom["count"]])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    table.wrapOn(c, width, height)
    cur_y -= 50
    table.drawOn(c, 100, cur_y)

    # Save the PDF
    c.save()

main()




