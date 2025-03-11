import os
import json
import csv
import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
from pdf2image import convert_from_path
import easyocr
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve and verify the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Please set OPENAI_API_KEY in your .env file.")

# OpenAI Client Initialization
client = OpenAI(api_key=api_key)

# Initialize EasyOCR Reader
reader = easyocr.Reader(["en"])  # Supports multiple languages

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

# Function to extract tables from a PDF file
def extract_tables_from_pdf(pdf_path):
    tables_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_table = page.extract_table()
                if extracted_table:
                    df = pd.DataFrame(extracted_table)
                    tables_data.append(df.to_json(orient="records"))
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
    return tables_data

# Function to extract images from a PDF file
def extract_images_from_pdf(pdf_path, output_dir="extracted_images"):
    os.makedirs(output_dir, exist_ok=True)
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_number in range(len(pdf_document)):
            for img_index, img in enumerate(pdf_document[page_number].get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_p{page_number + 1}_img{img_index}.png")
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                images.append(image_path)
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {e}")
    return images

# Function to perform OCR on images using EasyOCR
def analyze_images_with_easyocr(images):
    image_descriptions = []
    for image_path in images:
        try:
            print(f"Processing image {image_path} with EasyOCR...")
            result = reader.readtext(image_path, detail=0)  # Extract text only
            extracted_text = " ".join(result) if result else "No text found."
            image_descriptions.append(f"Text from {os.path.basename(image_path)}: {extracted_text}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    return "\n".join(image_descriptions)

# Function to process extracted content using OpenAI
def process_pdf_with_openai(pdf_path):
    prompt = "Extract key insights from the given PDF content. Summarize the main topics, tables, and any extracted text from images if present."
    # Upload the user provided file to OpenAI
    message_file = client.files.create(
        file=open(pdf_path, "rb"), purpose="assistants"
    )

    messages = [
        {
            "role": "system", 
            "content": "You are an AI trained to summarize and extract structured data from PDFs, including tables with text if present."
        },
        {
            "role": "user",
            "content": prompt,
            # Attach the new file to the message.
            "attachments": [
                { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
            ],
        },
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        max_tokens=500
    )

    print(f"OpenAI response: {response}")  # Debug print to check the response

    return response.choices[0].message.content.strip()

# Function to process all PDFs in a directory and save output to a CSV file
def process_pdfs(input_directory, output_csv_file):
    pdf_files = [f for f in os.listdir(input_directory) if f.endswith(".pdf")]
    extracted_data = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_directory, pdf_file)
        print(f"Processing {pdf_file}...")
        openai_summary = process_pdf_with_openai(pdf_path)

        extracted_data.append({
            "file_name": pdf_file,
            "summary": openai_summary
        })

    # Convert extracted data to a CSV file
    df = pd.DataFrame(extracted_data)
    df.to_csv(output_csv_file, index=False, encoding="utf-8")
    print(f"CSV saved: {output_csv_file}")

# Main execution
if __name__ == "__main__":
    input_directory = "data"  # Update to your actual directory
    output_csv_file = "structured_extraction_output.csv"

    process_pdfs(input_directory, output_csv_file)