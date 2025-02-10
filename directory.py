import os

file_path = '/Users/lavonda/Documents/RAG/Aorta-follow-up.pdf'
if os.path.isfile(file_path):
    print("File exists and is accessible.")
else:
    print("File does not exist or the path is incorrect.")
