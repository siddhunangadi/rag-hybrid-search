def sliding_window(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = chunk_size - chunk_overlap
    pieces = []
    position = 0
    while position < len(text):
        pieces.append(text[position : position + chunk_size])
        position += step
    return pieces
