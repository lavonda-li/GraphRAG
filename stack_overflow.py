from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)
import os

filename = "data/Aorta-follow-up.pdf"
prompt = "Extract key insights from the given PDF content. Summarize the main topics, tables, and any extracted text from images if present."

client = OpenAI(api_key=os.environ.get("MY_OPENAI_KEY"))

pdf_assistant = client.beta.assistants.create(
    model="gpt-4o-mini-2024-07-18",
    description="An assistant to extract the contents of PDF files.",
    tools=[{"type": "file_search"}],
    name="PDF assistant",
)

# Create thread
thread = client.beta.threads.create()

file = client.files.create(file=open(filename, "rb"), purpose="assistants")

# Create assistant
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    attachments=[
        Attachment(
            file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")]
        )
    ],
    content=prompt,
)

# Run thread
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=pdf_assistant.id, timeout=1000
)

if run.status != "completed":
    raise Exception("Run failed:", run.status)

messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
messages = [message for message in messages_cursor]

# Output text
res_txt = messages[0].content[0].text.value
print(res_txt)