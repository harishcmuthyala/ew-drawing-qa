import fitz  # PyMuPDF for PDF processing
import json
import base64
import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Initialize the OpenAI model
api_key = "" # Replace with API Key
model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# Function to process a page and questions
def ask_questions_with_image(page_image, questions):
    # Convert the image to base64
    image_data = base64.b64encode(page_image).decode("utf-8")

    # Create a message with the image and questions
    messages = []
    for question in questions:
        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ]
        )
        messages.append(message)

    # Send the questions to the model
    responses = []
    for message in messages:
        response = model.invoke([message])
        responses.append(response.content)

    return responses

# Load the JSON file
with open("../docs/Drawing_Checklist_Structured.json", "r") as json_file:
    checklist_data = json.load(json_file)

# Open the PDF
pdf_file = "../docs/EWQA.pdf"
pdf_document = fitz.open(pdf_file)

# Prepare the output
output = {}

# Process each page in the checklist JSON
for page_number, questions in checklist_data.items():
    # Extract the corresponding page as an image
    page_index = int(page_number.split("-")[1]) - 1  # Convert page number to index
    page = pdf_document[page_index]
    pix = page.get_pixmap()
    page_image = pix.tobytes("jpeg")

    # Send the page image and questions to the LLM
    answers = ask_questions_with_image(page_image, questions)

    # Save the results
    output[page_number] = {"questions": questions, "answers": answers}

# Save the output to a new JSON file
output_file = "LLM_Responses.json"
with open(output_file, "w") as json_out:
    json.dump(output, json_out, indent=4)

print(f"Responses saved to {output_file}")
