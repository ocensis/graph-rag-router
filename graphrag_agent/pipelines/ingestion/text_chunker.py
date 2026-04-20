"""
文本分块器 —— 自动检测语种，中文用 jieba（原逻辑），英文用 LangChain RecursiveCharacterTextSplitter

为什么拆两条路:
  - jieba 的 `_is_sentence_end` 只认 '。！？'，对英文 paper 会退化成按固定长度切
  - LangChain Recursive 按 ["\n\n", "\n", ". ", "? ", "! ", " "] 分层优先，英文对齐更好
  - 实测 eng200 数据：英文下 Recursive 切到句中率 48% vs jieba 65%（-17pp）

输出格式一致（都是 List[List[str]]）以兼容 document_processor 下游:
  - jieba: 每个 chunk 是 token 列表, ''.join(chunk) 还原文本
  - Recursive: 每个 chunk 是单元素列表 [chunk_str], ''.join 不变
"""
import jieba
import re
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from graphrag_agent.config.settings import CHUNK_SIZE, OVERLAP, MAX_TEXT_LENGTH


# ---- 语种检测 ----
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _cjk_ratio(text: str, sample: int = 2000) -> float:
    """取前 sample 个字符看 CJK 占比（全文算太慢）"""
    if not text:
        return 0.0
    s = text[:sample]
    cjk = len(_CJK_RE.findall(s))
    return cjk / len(s) if s else 0.0


# ---- Token-to-char 换算 ----
# 英文: 1 token ≈ 4-5 chars (word tokenizer)
# 中文: 1 jieba token ≈ 1-2 chars
# CHUNK_SIZE 参数是 token 数，Recursive splitter 用 chars，需要换算
_EN_CHARS_PER_TOKEN = 5


class ChineseTextChunker:
    """
    保留原类名，内部按语种自动选分块策略
    （兼容所有上游调用；改了底层但接口 / 返回格式不变）
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP,
                 max_text_length: int = MAX_TEXT_LENGTH):
        if chunk_size <= overlap:
            raise ValueError("chunk_size必须大于overlap")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_text_length = max_text_length

        # 英文 splitter（字符单位），换算后的 size/overlap
        self._en_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * _EN_CHARS_PER_TOKEN,
            chunk_overlap=overlap * _EN_CHARS_PER_TOKEN,
            # 优先级：段落 > 行 > 句号 > 问号 > 感叹号 > 分号 > 逗号 > 空格 > 字符
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator=True,
        )

    def process_files(self, file_contents: List[Tuple[str, str]]) -> List[Tuple[str, str, List[List[str]]]]:
        results = []
        for filename, content in file_contents:
            chunks = self.chunk_text(content)
            results.append((filename, content, chunks))
        return results

    def chunk_text(self, text: str) -> List[List[str]]:
        """根据语种路由到 jieba 或 Recursive"""
        if not text or len(text) < self.chunk_size / 10:
            return [list(text)] if text else []

        cjk = _cjk_ratio(text)
        if cjk >= 0.20:
            # 中文为主: jieba 原逻辑
            return self._chunk_with_jieba(text)
        else:
            # 英文为主: LangChain Recursive
            return self._chunk_with_recursive(text)

    # ==================== English path (LangChain) ====================

    def _chunk_with_recursive(self, text: str) -> List[List[str]]:
        """英文用 RecursiveCharacterTextSplitter，返回 [[chunk_str], [chunk_str], ...]"""
        raw = self._en_splitter.split_text(text)
        # 兼容接口：每个 chunk 包一层 list；''.join([chunk_str]) 还是原文
        return [[c] for c in raw if c.strip()]

    # ==================== Chinese path (jieba 原逻辑) ====================

    def _preprocess_large_text(self, text: str) -> List[str]:
        if len(text) <= self.max_text_length:
            return [text]
        target_segment_size = min(self.max_text_length, max(10000, self.max_text_length // 2))
        paragraphs = text.split('\n\n')
        if len(paragraphs) < 5:
            paragraphs = text.split('\n')
        processed_segments = []
        current_segment = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > target_segment_size:
                if current_segment:
                    processed_segments.append(current_segment)
                    current_segment = ""
                split_paras = self._split_long_paragraph(para, target_segment_size)
                processed_segments.extend(split_paras)
            else:
                if len(current_segment) + len(para) + 2 > target_segment_size:
                    if current_segment:
                        processed_segments.append(current_segment)
                    current_segment = para
                else:
                    if current_segment:
                        current_segment += "\n\n" + para
                    else:
                        current_segment = para
        if current_segment:
            processed_segments.append(current_segment)
        return processed_segments

    def _split_long_paragraph(self, text: str, max_size: int) -> List[str]:
        if len(text) <= max_size:
            return [text]
        sentences = re.split(r'([。！？.!?])', text)
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence.strip():
                combined_sentences.append(sentence + punctuation)
        if not combined_sentences:
            return [text[i:i + max_size] for i in range(0, len(text), max_size)]
        segments = []
        current_segment = ""
        for sentence in combined_sentences:
            if len(sentence) > max_size:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = ""
                for i in range(0, len(sentence), max_size):
                    segments.append(sentence[i:i + max_size])
            else:
                if len(current_segment) + len(sentence) > max_size:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += sentence
        if current_segment:
            segments.append(current_segment)
        return segments

    def _safe_tokenize(self, text: str) -> List[str]:
        try:
            if len(text) > self.max_text_length:
                return list(text)
            tokens = list(jieba.cut(text))
            return tokens if tokens else []
        except Exception:
            return list(text)

    def _chunk_with_jieba(self, text: str) -> List[List[str]]:
        text_segments = self._preprocess_large_text(text)
        all_chunks = []
        for segment in text_segments:
            all_chunks.extend(self._chunk_single_segment(segment))
        return all_chunks

    def _chunk_single_segment(self, text: str) -> List[List[str]]:
        if not text:
            return []
        all_tokens = self._safe_tokenize(text)
        if not all_tokens:
            return []
        chunks = []
        start_pos = 0
        while start_pos < len(all_tokens):
            end_pos = min(start_pos + self.chunk_size, len(all_tokens))
            if end_pos < len(all_tokens):
                sentence_end = self._find_next_sentence_end(all_tokens, end_pos)
                if sentence_end <= start_pos + self.chunk_size + 100:
                    end_pos = sentence_end
            chunk = all_tokens[start_pos:end_pos]
            if chunk:
                chunks.append(chunk)
            if end_pos >= len(all_tokens):
                break
            overlap_start = max(start_pos, end_pos - self.overlap)
            next_sentence_start = self._find_previous_sentence_end(all_tokens, overlap_start)
            if next_sentence_start > start_pos and next_sentence_start < end_pos:
                start_pos = next_sentence_start
            else:
                start_pos = overlap_start
            if start_pos >= end_pos:
                start_pos = end_pos
        return chunks

    def _is_sentence_end(self, token: str) -> bool:
        return token in ['。', '！', '？']

    def _find_next_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        for i in range(start_pos, len(tokens)):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return len(tokens)

    def _find_previous_sentence_end(self, tokens: List[str], start_pos: int) -> int:
        for i in range(start_pos - 1, -1, -1):
            if self._is_sentence_end(tokens[i]):
                return i + 1
        return 0

    def get_text_stats(self, text: str) -> dict:
        cjk = _cjk_ratio(text)
        return {
            "length": len(text),
            "cjk_ratio": cjk,
            "language": "zh" if cjk >= 0.20 else "en",
        }
