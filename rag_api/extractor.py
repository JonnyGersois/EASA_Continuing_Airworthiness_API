# extractor.py
from typing import Optional
from lxml import etree
from langchain_core.documents import Document

NAMESPACES = {
    "er": "http://www.easa.europa.eu/erules-export",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def extract_clean_xml_from_package(xml_path: str, save_clean_path: Optional[str] = None) -> str:

    """
    For eRules Export XML, the file is already clean.
    We simply load it, normalise whitespace, and optionally save it.
    """

    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    tree = etree.parse(xml_path, parser)
    root = tree.getroot()

    clean_str = etree.tostring(root, pretty_print=True, encoding="unicode")

    if save_clean_path:
        with open(save_clean_path, "w", encoding="utf-8") as f:
            f.write(clean_str)

    return clean_str


def convert_xml_to_documents(clean_xml_str: str) -> list[Document]:
    """
    Convert eRules Export XML into LangChain Documents.
    Each <er:topic> becomes a Document with:
      - title (if present)
      - body text (from SDT-linked text or title-only fallback)
      - metadata (all attributes on <er:topic>)
    """

    parser = etree.XMLParser(remove_blank_text=True, recover=True)
    root = etree.fromstring(clean_xml_str.encode("utf-8"), parser=parser)

    docs = []

    # Build SDT map if w:sdt elements exist (some eRules exports include them)
    sdt_map = {}
    for sdt in root.xpath(".//w:sdt", namespaces=NAMESPACES):
        w_id = sdt.find(".//w:sdtPr/w:id", namespaces=NAMESPACES)
        if w_id is None:
            continue

        sdt_key = w_id.attrib.get(f"{{{NAMESPACES['w']}}}val")
        texts = [t.text for t in sdt.findall(".//w:t", namespaces=NAMESPACES) if t.text]

        if texts:
            sdt_map[sdt_key] = " ".join(texts)

    # Process all topics
    for topic in root.xpath(".//er:topic", namespaces=NAMESPACES):
        metadata = {k: v for k, v in topic.attrib.items()}

        # Title
        title_el = topic.find(".//er:title", namespaces=NAMESPACES)
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        # Body text from SDT if available
        sdt_id = topic.attrib.get("sdt-id")
        body_text = sdt_map.get(sdt_id, "")

        # Fallback: if no SDT text, use title only
        content = (title + "\n\n" + body_text).strip() if body_text else title

        docs.append(Document(page_content=content, metadata=metadata))

    return docs