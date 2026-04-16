from pypdf import PdfReader
from typing import List


def load_pdf(file_path: str) -> str:
    """
    Load text from a PDF file
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def load_text(file_path: str) -> str:
    """
    Load text from a .txt file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents(file_path: str) -> List[str]:
    """
    Load document(s) and return as list.

    Args:
        file_path (str): Path to file

    Returns:
        List[str]: List of documents
    """
    if file_path.endswith(".pdf"):
        text = load_pdf(file_path)
    elif file_path.endswith(".txt"):
        text = load_text(file_path)
    else:
        raise ValueError("Unsupported file type")

    # Always return list (important for pipeline consistency)
    return [text]