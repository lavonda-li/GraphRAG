import csv
from openai import OpenAI

client = OpenAI()

assistant = client.beta.assistants.create(
    name="File Summary Assistant",
    instructions="Extract key insights from the given PDF content. Summarize the main topics, tables, and any extracted text from images if present.",
    model="gpt-4o-mini-2024-07-18",
    tools=[{"type": "file_search"}],
)

# Create a vector store called "Medical Protocols"
vector_store = client.beta.vector_stores.create(name="Medical Protocols")

filename = "data/Aorta-follow-up-copy.pdf"
prompt = "Extract key insights from the given PDF content. Summarize the main topics, tables, and any extracted text from images if present."

# Ready the files for upload to OpenAI
file_paths = [filename]  # Ensure this is a list
file_streams = [open(path, "rb") for path in file_paths]

# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
)

# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

# Upload the user-provided file to OpenAI
message_file = client.files.create(
    file=open(filename, "rb"), purpose="assistants"
)

# Create a thread and attach the file to the message
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
            # Attach the new file to the message.
            "attachments": [
                {"file_id": message_file.id, "tools": [{"type": "file_search"}]}
            ],
        }
    ]
)

# The thread now has a vector store with that file in its tool resources.
print(thread.tool_resources.file_search)

from typing_extensions import override
from openai import AssistantEventHandler

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call['type']}\n", flush=True)

    @override
    def on_message_done(self, message) -> None:
        # print a citation to the file searched
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        print("Message content:", message_content.value)
        print("Citations:", "\n".join(citations))

        # Save the output to a CSV file
        output_csv_file = "file_search_output.csv"
        with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["summary"])
            writer.writerow([message_content.value])

        print(f"CSV saved: {output_csv_file}")

# Then, we use the stream SDK helper
# with the EventHandler class to create the Run
# and stream the response.

with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account.",
    event_handler=EventHandler(),
) as stream:
    stream.until_done()