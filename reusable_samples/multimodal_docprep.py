###Multimodal_docprep.py###

import os
import io
import base64
import json
from pdf2image import convert_from_path
from openai import AzureOpenAI
import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel
from PyPDF2 import PdfReader
#local imports
from aoai import inference_structured_output_aoai, inference_aoai
from document_processing import download_blob

API_VERSION = "2024-08-01-preview"

# Azure OpenAI configuration
aoai_deployment = os.environ.get("AOAI_DEPLOYMENT_NAME")
aoai_key = os.environ.get("AOAI_API_KEY")
aoai_endpoint = os.environ.get("AOAI_ENDPOINT")


storage_account_conn_str = os.environ.get("STORAGE_ACCOUNT_CONNECTION_STRING")
storage_account_container = os.environ.get("STORAGE_ACCOUNT_CONTAINER")
storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")


# Initialize Azure OpenAI client
aoai_client = AzureOpenAI(
        azure_endpoint=aoai_endpoint,
        api_key=aoai_key,
        api_version=API_VERSION)

document_structure_analyzer_prompt = """You are responsible for analyzing a document to determine its topic and top-level sections.

Output the following fields:
Summary: A brief description of the document type and its purpose. Make sure to include who/what the main subjects are.
Top-level Sections: What are the top-level sections of the document? 


###Examples###
User: (a large document of a home inspection report)
Assistant: 
Summary: Home Inspection Report for 337 Goldman Drive, Phoenixville, PA 19460, the home of Dan Giannone.
Top-level Sections: General Information
Introduction and Structural Overview
Purpose and Scope
Roof System
Exteriors
Electrical System
Heating and Cooling
Fireplaces and Chimneys
Plumbing System
Interiors
Basement and Crawlspace
Attic Area and Roof Framing
Appliances
NACHI Standards of Practice

"""


image_prompt = """You will be given an image that is one or more pages of a document, along with some analysis of the overall document. 


###Output Structure###
{
text: The verbatim text of the page in markdown format. 
images: A 1 sentence description of any images on the page and how they relate to the text.
image_insights: All insights or information that can be gleaned from the images on the page and the relationship to the text. 
}

###Guidance###

1. Output 'na' for images or image_insights if there are no images or diagrams. Do not consider tables images as you will be capturing them via text.
2. When outputting markdown, keep in mind that you are only looking at one page of a much larger document, so only consider something a section header if you feel very confident it could be a section header for this type of document.
3. In the text, make sure to indicate where the images are located on the page with [] brackets.
4. Use the surrounding text to provide context to the image & extract further insights from it. For example, if the text describes a picture of a house with "ADDRESS" listed below it, you can assume the image of the house is that address. Be as descriptive as possible. Just explain, do not start with "the image is..."
5. Only use markdown H2 headers for the top-level sections mentioned in the document structure analysis. Everything else should be a H3 or lower or some other markdown element.

###Examples###

User: (an image of the following text & picture) 
<document analysis>
Summary: Home Inspection Report for 337 Goldman Drive, Phoenixville, PA 19460, the home of Dan Giannone.
top-level sections: General Information
Introduction and Structural Overview
Purpose and Scope
Roof System
Exteriors
Electrical System
Heating and Cooling
Fireplaces and Chimneys
Plumbing System
Interiors
Basement and Crawlspace
Attic Area and Roof Framing
Appliances
NACHI Standards of Practice

<Content>
LDS Inspections
A Division of Lennox Design Studios

2801 Soni Drive Trooper, PA 19403
Phone: 610-277-4953 Fax: 610-277-4954
WWW.LDSINSPECTIONS.COM

Home Inspection Report For

---

(Image of a house)

337 Goldman Drive
Phoenixville, PA 19460

---

Report Prepared For
Dan Giannone

Report Prepared By
Craig Lennox


Assistant:

text: 
# LDS Inspections
**A Division of Lennox Design Studios**

2801 Soni Drive Trooper, PA 19403  
Phone: 610-277-4953 Fax: 610-277-4954  
[WWW.LDSINSPECTIONS.COM](http://www.ldsinspections.com)

**Home Inspection Report For**

---

[Image]

**337 Goldman Drive  
Phoenixville, PA 19460**

---

*Report Prepared For*  
**Dan Giannone**

*Report Prepared By*  
**Craig Lennox**

image_insights: 337 Goldman Dr, a large two-story suburban house owned by Dan Giannone. The house has the following features:

White exterior with light blue or gray trim
Multiple peaked roof sections
Several windows, including some arched windows on the upper floor
Two-car garage with white doors
Paved driveway with two vehicles parked in it (appear to be dark-colored sedans or similar)
Well-maintained front lawn
Some landscaping, including a small tree or bush with reddish foliage near the front of the house
Part of a neighboring house visible on the left side
Clear blue sky visible

"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def pdf_to_base64_images(pdf_path, output_dir):
    pdf_document = fitz.open(pdf_path)
    base64_images = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_pages = len(pdf_document)

    for page_num in range(total_pages):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        temp_image_path = os.path.join(output_dir, f"page_{page_num}.png")
        img.save(temp_image_path, format="PNG")
        base64_image = encode_image(temp_image_path)
        base64_images.append(base64_image)
        os.remove(temp_image_path)  # Remove the temporary image file

    return base64_images

class OutputStructure(BaseModel):
    text: str
    image_insights: str

def analyze_document_structure(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    messages = [
        {"role": "system", "content": document_structure_analyzer_prompt},
        {"role": "user", "content": full_text}
    ]

    document_structure_analysis = inference_aoai(messages, aoai_deployment)
    return document_structure_analysis.choices[0].message.content

def process_image(image, page_num, source_filename, document_structure):
    messages = [
        {
            "role": "system",
            "content": f"{image_prompt}"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"source_filename: {source_filename}\npage_number: {page_num}\n\nDocument Structure Analysis:\n{document_structure}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}", "detail": "high"}}
            ]
        }
    ]
    
    raw_response = inference_structured_output_aoai(messages, aoai_deployment, OutputStructure)
    if raw_response:
        response = OutputStructure(**raw_response.choices[0].message.parsed.dict())
        print(f"Processed page {page_num}")
        #print(f"Text: {response.text}")
        print(f"Image Insights: {response.image_insights}")
        return response
    else:
        print(f"Failed to process page {page_num}")
        return None

def create_consolidated_markdown(processed_pages):
    consolidated_output = ""
    for page_num, page_data in enumerate(processed_pages, start=1):
        consolidated_output += page_data.text + "\n\n"
        if page_data.image_insights != 'na':
            consolidated_output += f"Image Insights: {page_data.image_insights}\n\n"
        consolidated_output += f"<Page {page_num}>\n\n"
        consolidated_output += "---\n\n"  # Add a separator between pages
    return consolidated_output

def main(input_path, filename):
    input_file = os.path.join(input_path, filename)
    
    # Analyze document structure
    document_structure = analyze_document_structure(input_file)
    print("Document Structure Analysis:")
    print(document_structure)
    
    # Create output folder based on the filename
    base_name = os.path.splitext(filename)[0]
    output_image_folder = os.path.join(input_path, f"{base_name}_images")
    
    # Create the folder if it doesn't exist
    os.makedirs(output_image_folder, exist_ok=True)
    
    # Convert PDF to images and save them in the output_image_folder
    base64_images = pdf_to_base64_images(input_file, output_image_folder)

    # Process each image and accumulate results in memory
    processed_pages = []
    for page_num, image in enumerate(base64_images, start=1):
        page_result = process_image(image, page_num, filename, document_structure)
        if page_result:
            processed_pages.append(page_result)

    # Create consolidated markdown content
    consolidated_markdown = create_consolidated_markdown(processed_pages)

    # Save the consolidated markdown to a file
    output_file = os.path.join(input_path, f"{base_name}_consolidated.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(consolidated_markdown)

    print(f"Processing complete. Consolidated results saved to {output_file}")


if __name__ == "__main__":
    # Hardcode the path and filename here
    input_path = "C:/temp/data/djg"
    filename = "337 Goldman Drive.pdf"
    temp_path = "C:/temp/data/djg/temp"
    container = 'djg'
    
    download_blob(filename, f'{temp_path}/{filename}', container)
    
    main(input_path, filename)