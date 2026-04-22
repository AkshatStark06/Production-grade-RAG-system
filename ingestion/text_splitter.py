import re


def split_text(documents, chunk_size=500, overlap=50):
    chunks = []

    for doc in documents:
        # Clean document
        doc = doc.strip()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', doc)

        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue  # ✅ skip empty sentences

            # If adding sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk (with overlap)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

    # ✅ Remove any empty chunks (safety)
    chunks = [c for c in chunks if c.strip()]

    return chunks