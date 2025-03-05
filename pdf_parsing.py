import csv
import os
from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)

# 1. Define your data folder and prompt
DATA_FOLDER = "fulldata"
PROMPT = (
    "Read the PDF, extract/parse key insights from the given PDF content. "
    "Summarize the main topics, tables (if any), and extracted text from images (if any), you don't need to mention tables/images if none present. Don't ask me any questions, just summarize the content."
)


# 2. Initialize OpenAI client using an environment variable for the API key
client = OpenAI(api_key=os.environ.get("MY_OPENAI_KEY"))

# 3. Create the specialized PDF assistant once (we can reuse the same assistant for all files)
pdf_assistant = client.beta.assistants.create(
    model="gpt-4o-mini-2024-07-18",
    description="An assistant to extract the contents of PDF files.",
    tools=[{"type": "file_search"}],
    name="PDF assistant",
)

# 4. Find all PDF files in the data folder
pdf_files = [
    f for f in os.listdir(DATA_FOLDER)
    if f.lower().endswith(".pdf")
]


# 5. Prepare a list to hold (filename, summary) pairs
pdf_summaries = []

# 6. Process each PDF file in a loop
for pdf_file in pdf_files:
    file_path = os.path.join(DATA_FOLDER, pdf_file)
    print(f"Processing {file_path}...")

    # Create a new thread for this PDF
    thread = client.beta.threads.create()

    # Upload the PDF
    file_upload = client.files.create(file=open(file_path, "rb"), purpose="assistants")

    # Create a user message with the PDF attachment
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        attachments=[
            Attachment(
                file_id=file_upload.id,
                tools=[AttachmentToolFileSearch(type="file_search")]
            )
        ],
        content=PROMPT,
    )

    # Run the thread and wait for completion
    thread_run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=pdf_assistant.id,
        timeout=1000
    )

    if thread_run.status != "completed":
        raise RuntimeError(f"Run failed for {pdf_file} with status: {thread_run.status}")

    # Grab all messages
    messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
    all_messages = list(messages_cursor)

    # Extract summary text from the assistant's response
    if not all_messages or not all_messages[0].content:
        raise RuntimeError(f"No assistant content returned for {pdf_file}.")
    summary_text = all_messages[0].content[0].text.value

    # Store the filename and summary together
    pdf_summaries.append((pdf_file, summary_text))

# 7. Write all summaries to a single CSV
output_csv_file = "parse_output.csv"
with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # Write a header row
    writer.writerow(["filename", "summary"])
    # Write one row per PDF
    for filename, summary in pdf_summaries:
        writer.writerow([filename, summary])

print(f"All PDF summaries saved to: {output_csv_file}")
