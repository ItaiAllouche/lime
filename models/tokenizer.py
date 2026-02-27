class SimpleCTCTokenizer:
    """
    small tokenizer for CTC.
    vocab:
      - id 0: blank
      - id 1..N: characters from `charset`
    """
    def __init__(self, charset: str = "-abcdefghijklmnopqrstuvwxyz '"):
        self.charset = charset
        self.blank_id = 0

        # map chars to ids: start from 1 (0 is blank)
        self.char2id = {ch: i + 1 for i, ch in enumerate(charset)}
        self.id2char = {i + 1: ch for i, ch in enumerate(charset)}
        self.vocab_size = len(charset) + 1  # +1 for blank
    
    def normalize(self, text: str):
        text = text.lower()
        
        # keep only chars we know; everything else → space
        normalized = [ch if ch in self.charset else " " for ch in text]
        # collapse multiple spaces
        return " ".join("".join(normalized).split())

    def encode(self, text: str):
        """return list of label IDs (no blanks; CTC handles blanks internally)."""
        text = self.normalize(text)
        return [self.char2id[ch] for ch in text if ch in self.char2id.keys()]

    def decode_ids(self, ids: list[int]):
        """decode a sequence of IDs (no CTC collapsing here)."""
        return "".join(self.id2char.get(i, "") for i in ids if i != self.blank_id)