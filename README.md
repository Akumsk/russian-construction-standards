# Russian Construction Dataset

📚 An open metadata dataset of widely used Russian construction standards.

This repository provides structured JSON metadata for Russian construction norms and rules (e.g., СП, СНиП, ГОСТ), without distributing the full PDF documents themselves. It is designed to be a collaborative, searchable knowledge base for engineers, researchers, and developers working with Russian regulatory standards.

⚠️ Note: The dataset includes only plain text content extracted either directly or via a simple OCR process. It does not contain any graphical elements such as illustrations, engineering diagrams, or formatting structures. Mathematical formulas and technical schemas are also extracted as plain text and may lack their original visual representation.

---

## 🗂 Repository Structure

```
russian-construction-dataset/
├── metadata/
│   ├── SP_28.13330.2017.json
│   └── ...
├── code/
│   ├── create_dataset.py
│   ├── check_document_status.py
│   └── ...
├── pdf_docs/
│   └── ...
├── LICENSE
├── LICENSE_METADATA.md
└── README.md
```


## 🤝 Contributing

We welcome and encourage community contributions!
If you'd like to add a new document metadata file, you have two options:

### 📄 1. Manual Submission
- Generate a JSON file from the original PDF using the methods provided in the `code/` folder.
- Fork this repository.
- Add your JSON file to the `metadata/` directory.
- Submit a pull request with a brief description of your changes.

### 📤 2. Upload Only
- Simply upload the original PDF file of the Russian construction standard in `pdf_docs/` folder.
- Our team will handle the extraction and metadata creation on your behalf.
All submissions are reviewed before being merged.
---

### 🛠️ OCR & Extraction Improvements
We also encourage contributions to enhance the OCR extraction process, including:

- Improving text recognition accuracy.
- Extracting tables and structured data.
- Converting mathematical formulas to LaTeX or other structured formats.
- Preserving diagrams, schemas, and visual layouts as accurately as possible.

Your support helps make this project more comprehensive and valuable to the community.

## ⚠️ Disclaimer
This repository contains only metadata describing Russian construction standards (e.g., title, number, category, issuance date, etc.) and does not include the original PDF documents.
All referenced standards are the intellectual property of their respective organizations or government bodies. This project does not claim ownership of the underlying documents. Users are responsible for ensuring legal compliance when accessing or using the full versions of these documents from official sources.

## 📜 License
Code in this repository is licensed under the MIT License.
Metadata files (.json) are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).


## 📄 JSON Metadata Structure

Each standard is described in a JSON file with the following structure:

```json
{
  "filename": "Свод правил 28.13330.2017.pdf",
  "full_name": "СП 28.13330.2017 «СНиП 2.03.11-85 Защита строительных конструкций от коррозии»",
  "number": "28.13330.2017",
  "date_issue": "2017-08-28",
  "document_type": "Свод правил",
  "language": "Russian",
  "category": "Конструкции",
  "source": "СП 28.13330.2017.pdf",
  "total_pages": 118,
  "status": "Действует",
  "pages": [
    {
      "page": 1,
      "page_content": "..."
    },
    {
      "page": 2,
      "page_content": "..."
    }
  ]
}


