#pdf_parser.py
import pdfplumber
import httpx
import tempfile

async def download_pdf(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            print(f"✅ Downloaded PDF to: {temp_file.name}")
            return temp_file.name

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            else:
                print(f"⚠️ No text found on page {i+1}")

    chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]
    
    print(f"📄 Extracted {len(chunks)} chunks from PDF using pdfplumber")
    print(f"🔍 First chunk (100 chars): {repr(chunks[0][:100]) if chunks else 'No chunks'}")
    
    return chunks
