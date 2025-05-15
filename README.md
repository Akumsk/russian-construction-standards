# Russian Construction Dataset

📚 An open metadata dataset of widely used Russian construction standards.

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
We welcome community contributions! To submit a new document metadata file:
Fork the repository.
Add your JSON file in the metadata/ folder.
Make sure it follows the structure and field naming convention.
Submit a pull request. All contributions are reviewed before approval.

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


