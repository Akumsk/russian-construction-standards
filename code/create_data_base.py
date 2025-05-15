#create_dataset.py

import os
import re
import json
import fitz
import logging
import time
from tqdm import tqdm
from PIL import Image, ImageOps, ImageFilter
import openai
import check_document_status

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_NAME = "gpt-4.1-mini"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI API key from settings
openai.api_key = OPENAI_API_KEY

###SETTINGS
rotate_angle = 0
count_of_pages_for_analysis = 5
enable_fast_extraction = True     # Use fast text extraction where possible
perform_ocr_on_no_text_only = True # Only perform OCR if no text found with fast extraction
skip_empty_pages = True           # Skip pages that have no text and no images
validate_document_status = True   # Check document status online

def preprocess_image(image, pdf_file_path=None):
    """
    Preprocess the PIL image for better OCR accuracy.
    Upscales the image, converts to grayscale, enhances contrast,
    applies a median filter, and then uses adaptive thresholding if possible.

    Parameters:
    - image (Image.Image): The PIL image to process.
    - pdf_file_path (str, optional): The full path to the PDF file. Defaults to None.

    Returns:
    - Image.Image: The final processed image.
    """
    try:
        # Check if image is valid
        if not image or image.size[0] <= 0 or image.size[1] <= 0:
            logger.warning("Invalid image size or empty image")
            return None

        # Upscale the image
        upscaled_img = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        # Convert to grayscale
        gray_image = ImageOps.grayscale(upscaled_img)
        # Enhance contrast
        enhanced_img = ImageOps.autocontrast(gray_image)
        # Apply a median filter to reduce noise
        filtered_img = enhanced_img.filter(ImageFilter.MedianFilter(size=3))

        # Try adaptive thresholding with OpenCV, fallback to simple thresholding if not available
        try:
            import cv2
            import numpy as np
            np_img = np.array(filtered_img)
            # Adaptive thresholding using Gaussian method
            thresh_img = cv2.adaptiveThreshold(
                np_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            final_img = Image.fromarray(thresh_img)
        except ImportError:
            # Fallback: apply a simple binary thresholding
            threshold = 70
            final_img = filtered_img.point(lambda x: 0 if x < threshold else 255, '1').convert("L")

        # Rotate image if needed
        final_img = final_img.rotate(angle=rotate_angle, expand=True)

        # Save the processed image if path is provided
        if pdf_file_path:
            folder = os.path.dirname(pdf_file_path)
            base = os.path.splitext(os.path.basename(pdf_file_path))[0]
            # Build a unique filename to prevent overwriting
            save_filename = f"{base}_improved.png"
            save_path = os.path.join(folder, save_filename)
            idx = 1
            while os.path.exists(save_path):
                save_filename = f"{base}_improved_{idx}.png"
                save_path = os.path.join(folder, save_filename)
                idx += 1
            final_img.save(save_path)
            logger.info(f"Saved improved image to {save_path}")

        return final_img
    except Exception as e:
        logger.warning(f"Error preprocessing image: {e}")
        return None


class OCRImageParser:
    def __init__(self, lang='ru'):
        """
        Initialize the OCR parser using EasyOCR.

        Parameters:
        - lang (str): Language code (e.g., 'ind' for Indonesian is converted to 'id').
        """
        if lang.lower() == 'ind':
            lang = 'id'
        self.lang = lang
        import easyocr  # local import to avoid dependency issues
        self.reader = easyocr.Reader([lang], gpu=False) #gpu=True for GPU processing

    def parse(self, image_blob):
        """
        Parse the image blob using EasyOCR after preprocessing.
        """
        try:
            # Convert the image blob to a PIL Image
            import io
            from PIL import Image, UnidentifiedImageError

            try:
                image = Image.open(io.BytesIO(image_blob))
            except UnidentifiedImageError:
                logger.warning("Cannot identify image file - possibly corrupted or empty")
                return ""
            except Exception as img_open_err:
                logger.warning(f"Failed to open image: {img_open_err}")
                return ""

            processed_image = preprocess_image(image)
            if processed_image is None:
                return ""

            # Convert processed image to a numpy array for EasyOCR
            import numpy as np
            result = self.reader.readtext(np.array(processed_image), detail=0)
            text = " ".join(result) if result else ""
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR parsing error: {e}")
            return ""


def get_document_info_from_llm(text, file_name):
    """
    Use LLM to analyze the document text and suggest proper metadata.

    Args:
        text (str): Text extracted from the document
        file_name (str): Original file name

    Returns:
        dict: Document metadata
    """
    # Define function schema for the LLM
    functions = [
        {
            "name": "extract_metadata",
            "description": "Extract comprehensive metadata from a document text sample. The metadata will be used to categorize and organize documents in a knowledge base system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {
                        "type": "string",
                        "description": (
                            "The complete title of the document in its original language. "
                            "Examples: 'Системы противопожарной защиты электроустановки низковольтные. Требования пожарной безопасности', "
                            "'Ограждения металлические лестниц, балконов, крыш, лестничных маршей и площадок. Общие технические условия', "
                            "'Bетонные и железобетонные конструкции. Основные положения'"
                        )
                    },
                    "number": {
                        "type": "string",
                        "description": (
                            "The document's official reference number or identifier code. "
                            "Examples: '462.1325800.2019', "
                            "'317.1325800.2017', "
                            "'10922-2012', "
                            "'123'"
                        )
                    },
                    "date_issue": {
                        "type": "string",
                        "description": (
                            "The latest date of publication, issue or update of the document in any consistent format. "
                            "If multiple dates are present, ALWAYS choose the most recent date. "
                            "Check carefully for update dates, revisions, or amendments that may be more recent than the original publication date. "
                            "For Russian documents, look for 'Дата введения', 'Дата актуализации', 'Дата введения изменения', 'с изменениями', or similar phrases. "
                            "Examples: '2019', "
                            "'2017-01', "
                            "'2020-01-01'"
                        )
                    },
                    "type": {
                        "type": "string",
                        "description": (
                            "The high-level classification of the document type. Always provide result in original language. "
                            "Examples:  'Свод правил', 'ГОСТ', 'СанПин', 'Технический регламент', 'Пособие', 'Федеральный закон', "
                            "'Кодекс', 'Постановление', 'Указ', 'Приказ', 'Пояснение', 'Правила."
                        )
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "The construction or design category that this document pertains to. Always provide result in the original language. "
                            "Examples: 'Архитектура', 'Конструкции', 'Отопление', 'Вентиляция', 'Водоснабжение', 'Канализация', "
                            "'Электроосвещение', 'Электроснабжение', 'Электрооборудование', 'Слаботочные системы', 'Связь', 'Автоматизация', "
                            "'Диспетчеризация', 'Газоснабжение', 'Организация строительства', 'Пожарная безопасность', 'Экология', 'Геодезия', "
                            "'Геология', 'Акустика', 'Энергоэффективность' и тому подобное"
                        )
                    },
                    "language": {
                        "type": "string",
                        "description": (
                            "The primary language of the provided document text. "
                            "Examples: 'English', 'French', 'Spanish', 'Indonesian', 'Russian'"
                        )
                    }
                },
                "required": [
                    "full_name",
                    "number",
                    "date_issue",
                    "type",
                    "category",
                    "language"
                ]
            }
        }
    ]

    try:
        # Create the system and user messages for the ChatCompletion
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant specialized in extracting structured metadata from document text samples. "
                    "Extract the document's full name, number, date of issue, type, category, and language. "
                    "Keep all extracted text in the original language. "
                    "For date_issue, always extract the LATEST date of publication, updates or revisions - not the original publication date if newer versions exist. "
                    "Ensure the extracted information accurately reflects the provided text sample. "
                    "Always respond by calling the function 'extract_metadata' with the extracted arguments."
                )
            },
            {
                "role": "user",
                "content": f"Analyze this document text sample and extract its metadata:\n\n{text}"
            }
        ]

        # Call the OpenAI ChatCompletion with function calling
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            functions=functions,
            function_call={"name": "extract_metadata"},  # Force the model to call our function
            temperature=0  # More deterministic
        )

        # Extract the function call arguments
        function_call = response.choices[0].message.function_call
        arguments_str = function_call.arguments

        # Parse the arguments
        parsed_args = json.loads(arguments_str)

        # Return the metadata
        return parsed_args

    except Exception as e:
        logger.error(f"Error getting document info from LLM: {e}")
        return None


def extract_pdf_content(pdf_file_path, ocr_process=False):
    """
    Extract content from a PDF file, with optional OCR processing.

    Args:
        pdf_file_path (str): Path to the PDF file
        ocr_process (bool): Whether to use OCR for text extraction

    Returns:
        dict: Dictionary containing extracted pages and content
    """
    try:
        doc = fitz.open(pdf_file_path)
        total_pages = doc.page_count
        extracted = {
            "filename": os.path.basename(pdf_file_path),
            "source": os.path.basename(pdf_file_path),  # Keep source at document level
            "total_pages": total_pages,
            "pages": []
        }

        # Initialize OCR parser if needed
        images_parser = None
        if ocr_process:
            # Determine language from filename (you might want to enhance this)
            lang = 'ru'  # Default to Russian
            if 'indonesian' in pdf_file_path.lower():
                lang = 'ind'
            images_parser = OCRImageParser(lang=lang)

        # Check if we can use fast extraction
        extraction_start_time = time.time()

        # Process each page
        for page_number in tqdm(range(total_pages), desc=f"Processing pages in {os.path.basename(pdf_file_path)}"):
            try:
                page = doc.load_page(page_number)
                text = ""

                # Fast text extraction - try first if enabled
                if enable_fast_extraction:
                    start_time = time.time()
                    raw_text = page.get_text()
                    text = raw_text.strip() if raw_text else ""
                    extraction_time = time.time() - start_time
                    if text:
                        logger.debug(f"Fast text extraction successful for page {page_number+1} in {extraction_time:.2f}s")

                # Check if we need to perform OCR
                need_ocr = False
                if ocr_process:
                    if not text and perform_ocr_on_no_text_only:
                        # No text found with fast extraction, try OCR
                        need_ocr = True
                        logger.debug(f"No text found with fast extraction for page {page_number+1}, trying OCR")
                    elif not perform_ocr_on_no_text_only:
                        # Always perform OCR if not limited to empty text pages
                        need_ocr = True

                # If OCR is needed and we have an OCR parser
                if need_ocr and images_parser:
                    ocr_texts = []
                    ocr_start_time = time.time()

                    try:
                        # First try to get embedded images
                        image_list = page.get_images(full=True)

                        if image_list:
                            for img in image_list:
                                try:
                                    xref = img[0]
                                    base_image = doc.extract_image(xref)
                                    if base_image and "image" in base_image:
                                        image_bytes = base_image.get("image")
                                        if image_bytes:
                                            try:
                                                ocr_result = images_parser.parse(image_bytes)
                                                if ocr_result:
                                                    ocr_texts.append(ocr_result)
                                            except Exception as img_err:
                                                logger.debug(f"OCR error on embedded image in page {page_number+1}: {img_err}")
                                except Exception as ext_err:
                                    logger.debug(f"Error extracting image from page {page_number+1}: {ext_err}")
                                    continue

                        # If no embedded images or they failed, render the page as an image
                        if not ocr_texts and not text:
                            try:
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                                image_bytes = pix.tobytes("png")
                                if image_bytes:
                                    try:
                                        # Verify image data is valid before OCR
                                        from PIL import Image
                                        import io
                                        Image.open(io.BytesIO(image_bytes)).verify()  # This will throw an error if image is invalid

                                        ocr_result = images_parser.parse(image_bytes)
                                        if ocr_result:
                                            ocr_texts.append(ocr_result)
                                    except Exception as render_ocr_err:
                                        logger.debug(f"OCR error on rendered page {page_number+1}: {render_ocr_err}")
                            except Exception as render_err:
                                logger.debug(f"Error rendering page {page_number+1} as image: {render_err}")

                    except Exception as page_img_err:
                        logger.debug(f"Error processing images on page {page_number+1}: {page_img_err}")

                    ocr_time = time.time() - ocr_start_time
                    if ocr_texts:
                        logger.debug(f"OCR processing for page {page_number+1} completed in {ocr_time:.2f}s")
                        # If OCR produced any results, combine them with any existing text
                        text = "\n".join([text] + ocr_texts) if text else "\n".join(ocr_texts)

                # Check if we should skip empty pages
                if not text and skip_empty_pages:
                    # Check if page has any content (images, annotations, etc.)
                    has_images = len(page.get_images(full=False)) > 0
                    has_content = has_images or text

                    if not has_content:
                        logger.debug(f"Skipping empty page {page_number+1}")
                        continue

                # Always provide at least an empty string for pages with no content
                if not text:
                    text = " "

                # Add page data WITHOUT source field (as per requirement)
                page_data = {
                    "page": page_number + 1,
                    "page_content": text
                    # source field removed as per requirement
                }
                extracted["pages"].append(page_data)

            except Exception as page_err:
                logger.warning(f"Error processing page {page_number+1} in {pdf_file_path}: {page_err}")
                # Add an empty page entry to maintain page count integrity
                page_data = {
                    "page": page_number + 1,
                    "page_content": " "
                    # source field removed as per requirement
                }
                extracted["pages"].append(page_data)

        extraction_total_time = time.time() - extraction_start_time
        logger.info(f"Extracted {len(extracted['pages'])} pages from {pdf_file_path} in {extraction_total_time:.2f}s")
        doc.close()
        return extracted

    except Exception as e:
        logger.error(f"Error processing PDF file {pdf_file_path}: {e}")
        return None


def create_dataset(folder_path=None, file_path=None, ocr_process=False):
    """
    Create a dataset by extracting metadata and content from PDF files,
    renaming files based on document numbers, and saving JSON data.

    Args:
        folder_path (str, optional): Path to folder containing PDF files
        file_path (str, optional): Path to a single PDF file
        ocr_process (bool): Whether to use OCR for text extraction

    Returns:
        list: List of processed document data
    """
    logger.info("Starting dataset creation process")

    # Determine files to process
    files_to_process = []
    if file_path:
        if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
            logger.info(f"Processing single file: {file_path}")
            files_to_process = [file_path]
        else:
            logger.error(f"File does not exist or is not a PDF: {file_path}")
            return []
    elif folder_path:
        if os.path.isdir(folder_path):
            logger.info(f"Processing folder: {folder_path}")
            files_to_process = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith('.pdf')
            ]
        else:
            logger.error(f"Folder does not exist: {folder_path}")
            return []
    else:
        logger.error("No folder_path or file_path provided")
        return []

    # Process each file
    processed_documents = []
    for current_file in tqdm(files_to_process, desc="Processing PDF files"):
        logger.info(f"Processing file: {os.path.basename(current_file)}")

        try:
            # Step 1: Extract initial text for metadata analysis
            logger.info(f"Extracting initial text for metadata analysis from: {os.path.basename(current_file)}")
            extraction_start_time = time.time()
            pdf_document = fitz.open(current_file)
            total_pages = len(pdf_document)

            # Extract text from first pages only (limited by count_of_pages_for_analysis setting)
            pages_to_extract = min(count_of_pages_for_analysis, total_pages)
            content = ""
            for page_num in range(pages_to_extract):
                page = pdf_document[page_num]
                # Fast extraction for metadata
                page_text = page.get_text()
                if page_text:
                    content += page_text + "\n\n"

            # Limit content size to prevent token limit issues
            content = content[:6000]
            pdf_document.close()
            extraction_time = time.time() - extraction_start_time
            logger.info(f"Initial text extraction completed in {extraction_time:.2f}s")

            # Step 2: Get metadata using LLM
            logger.info(f"Extracting metadata for: {os.path.basename(current_file)}")
            metadata = get_document_info_from_llm(content, os.path.basename(current_file))

            if not metadata:
                logger.warning(f"Failed to extract metadata for: {os.path.basename(current_file)}")
                continue

            # Clean and standardize number - only keep digits, dots, and hyphens
            original_number = metadata["number"]
            metadata["number"] = clean_document_number(original_number)
            logger.info(f"Cleaned document number from '{original_number}' to '{metadata['number']}'")

            # Standardize hyphens in metadata
            metadata["full_name"] = standardize_hyphens(metadata["full_name"])

            # Step 3: Extract full content with OCR if requested
            logger.info(f"Extracting content for: {os.path.basename(current_file)}")
            content_data = extract_pdf_content(current_file, ocr_process)

            if not content_data:
                logger.warning(f"Failed to extract content for: {os.path.basename(current_file)}")
                continue

            # Step 4: Online status validation - only use check_document_status now
            document_status = "Неопределен"  # Default status
            if validate_document_status:
                try:
                    logger.info(f"Getting document status online for: {metadata['number']}")
                    # Use the document number (without prefixes) to check status online
                    online_status = check_document_status(metadata["number"], headless=True)
                    if online_status:
                        logger.info(f"Online status for '{metadata['number']}': {online_status}")
                        document_status = online_status
                    else:
                        logger.warning(f"Could not determine status online for '{metadata['number']}', using default: {document_status}")
                except Exception as e:
                    logger.error(f"Error checking document status online: {e}")

            # Step 5: Combine metadata and content
            document_data = {
                "filename": f"{metadata['type']} {metadata['number']}.pdf",
                "full_name": metadata["full_name"],
                "number": metadata["number"],
                "date_issue": metadata["date_issue"],
                "document_type": metadata["type"],
                "language": metadata["language"],
                "category": metadata["category"],
                "source": os.path.basename(current_file),  # Keep source at document level
                "total_pages": total_pages,
                "status": document_status,
                "pages": content_data["pages"]
            }

            # Step 6: Rename the PDF file based on document number
            file_dir = os.path.dirname(current_file)
            new_filename = f"{metadata['type']} {metadata['number']}.pdf"
            new_file_path = os.path.join(file_dir, new_filename)

            # Handle case where the new filename already exists
            counter = 1
            while os.path.exists(new_file_path) and new_file_path != current_file:
                new_filename = f"{metadata['type']} {metadata['number']}_{counter}.pdf"
                new_file_path = os.path.join(file_dir, new_filename)
                counter += 1

            # Rename the file
            if current_file != new_file_path:
                os.rename(current_file, new_file_path)
                logger.info(f"Renamed: {os.path.basename(current_file)} → {new_filename}")
                document_data["filename"] = new_filename

            # Step 7: Save JSON file with the same name
            json_filename = os.path.splitext(new_filename)[0] + ".json"
            json_file_path = os.path.join(file_dir, json_filename)

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(document_data, json_file, ensure_ascii=False, indent=4)

            logger.info(f"Saved JSON data to: {json_filename}")

            processed_documents.append(document_data)

        except Exception as e:
            logger.error(f"Error processing {os.path.basename(current_file)}: {e}")

    logger.info(f"Dataset creation completed. Processed {len(processed_documents)} documents")
    return processed_documents


def standardize_hyphens(text):
    """
    Replace various hyphen types with standard hyphen.

    Args:
        text (str): Text to standardize

    Returns:
        str: Text with standardized hyphens
    """
    if not text:
        return text

    # Replace various types of hyphens/dashes with standard hyphen
    # en dash (–), em dash (—), figure dash (‒), etc.
    hyphen_types = [
        '\u2010', '\u2011', '\u2012', '\u2013', '\u2014',
        '\u2015', '\u2212', '\u2043', '\u02D7', '\u058A',
        '\u05BE', '\u1400', '\u1806', '\u2E17', '\u2E1A',
        '\u2E3A', '\u2E3B', '\u2E40', '\u30FB', '\uFE58',
        '\uFE63', '\uFF0D'
    ]

    result = text
    for hyphen in hyphen_types:
        result = result.replace(hyphen, '-')

    return result


def clean_document_number(number):
    """
    Clean document number to include just digits, dots, and hyphens.
    Removes prefixes like "СП", "ГОСТ", etc.

    Args:
        number (str): Document number to clean

    Returns:
        str: Cleaned document number
    """
    if not number:
        return number

    # Standardize hyphens
    number = standardize_hyphens(number)

    # Define common document prefixes to remove
    prefixes = [
        'СП', 'ГОСТ', 'ГОСТ Р', 'СанПиН', 'СНиП', 'ТР', 'РД',
        'МДС', 'ВСН', 'ОДМ', 'ФЗ', 'ГН', 'СО', 'ТСН', 'НПБ'
    ]

    # Sort prefixes by length in descending order to match longer prefixes first
    prefixes.sort(key=len, reverse=True)

    cleaned = number.strip()

    # Try to find and remove prefixes
    for prefix in prefixes:
        # Check for exact prefix match followed by space or non-word character
        pattern = rf'^{re.escape(prefix)}\s+|\s+{re.escape(prefix)}\s+'
        cleaned = re.sub(pattern, ' ', cleaned).strip()

    # Extract the number pattern - typically digits, dots, and hyphens
    # This regex looks for patterns like 35.13330.2011 or 123-ФЗ
    match = re.search(r'([\d\.\-]+(?:\.\d+)*)', cleaned)
    if match:
        cleaned = match.group(1).strip()

    return cleaned


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create dataset from PDF documents')
    parser.add_argument('--folder', '-f', type=str, help='Path to folder containing PDF files')
    parser.add_argument('--file', type=str, help='Path to a single PDF file')
    parser.add_argument('--ocr', action='store_true', help='Enable OCR processing')

    args = parser.parse_args()

    if not args.folder and not args.file:
        print("Please provide either a folder path or a file path")
    else:
        create_dataset(folder_path=args.folder, file_path=args.file, ocr_process=args.ocr)

#Example usage:
#file_path = r'your file_path.pdf'
folder_path = r'pdf_docs'
#Create dataset from folder
create_dataset(folder_path=folder_path, file_path=None, ocr_process=True)
#Create dataset from file
#create_dataset(folder_path=None, file_path=file_path, ocr_process=True)

