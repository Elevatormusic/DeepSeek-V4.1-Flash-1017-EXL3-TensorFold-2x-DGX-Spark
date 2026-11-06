"""Reject complete model tool calls that were not offered for a request."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import uuid
from typing import Any


class ModelOutputError(Exception):
    """A safe public error for a model attempting an unavailable tool."""

    code = "unavailable_tool_call"

    def __init__(self, tool_name: str | None = None, *, diagnostic: dict | None = None):
        self.rejection_id = uuid.uuid4().hex[:16]
        reason = (diagnostic or {}).get("reason", "unavailable_tool")
        message = ("The model attempted to call a tool not offered by this request. "
                   f"Reason: {reason}; source: tensorfold.tool_output_guard; diagnostic ID: {self.rejection_id}.")
        super().__init__(message)
        self.error = {"type": "model_output_error", "code": self.code, "message": message,
                      "source": "tensorfold.tool_output_guard", "reason": reason, "rejection_id": self.rejection_id}
        self.tool_name = tool_name
        self.diagnostic = {**(diagnostic or {}), "rejection_id": self.rejection_id,
                           "source": self.error["source"], "code": self.code,
                           "tool_name": (tool_name or "")[:96]}

    def log_line(self) -> str:
        return f"[tensorfold] model output rejected: {self.code} " + json.dumps(self.diagnostic, ensure_ascii=True)


class _RunawayMarkup(Exception):
    """A call block whose Qwen or GLM form the parser would scan for long (looping output)."""


class ToolOutputGuard:
    """Validate complete call envelopes before the CUDA parser can execute any call."""

    _opener_re = re.compile(r"(?i:<tool_call>)|<｜DSML｜ calls>")
    # The parser's expressions try each opener against the rest of the text. Markup that stays open this often is
    # looping output, not a call: the guard rejects it before the parser has to scan it (and so does not stall).
    _RUNAWAY = 64

    _white = re.compile(r"\s+")
    _SCAN_BUDGET = 200_000_000    # a fixed limit on the estimated scan steps (see _stalls), not a performance claim
    _qwen_head = re.compile(r"^\s*<function=([^>\s]+)>", re.IGNORECASE)
    _qwen_close = re.compile(r"</function>\s*\Z", re.IGNORECASE)
    # The characters that the parser's case-insensitive expressions match with an ASCII letter (see the ``re`` docs).
    _fold_table = {0x130: "i", 0x131: "i", 0x17f: "s", 0x212a: "k"}

    @classmethod
    def _fold(cls, text: str) -> str:
        """Lower case as the parser's ``re.IGNORECASE`` sees it, with the same length as ``text``."""

        return text.translate(cls._fold_table).lower()

    @staticmethod
    def _open_tail(text: str, opener: str, closer: str) -> int:
        """How many ``opener`` have no ``closer`` after them."""

        return text.count(opener, text.rfind(closer) + 1)

    @classmethod
    def _stalls(cls, text: str, opener: str, closer: str) -> bool:
        """Whether the parser expressions could scan this text for long. An ``opener`` that no ``closer`` follows sends
        a lazy expression to the end of the text. With many of them the scan is quadratic. With white space it is worse:
        the expressions try the white space at each position, so the cost grows with the length of the text and with
        the square of its longest white-space run. The estimate below bounds that cost; it is far above any real call."""

        rest = text[text.rfind(closer) + 1:]
        count = rest.count(opener)
        if count == 0:
            return False
        if count > cls._RUNAWAY:
            return True
        # (a text without white space still costs one scan per opener: that counts as a run of 2)
        longest = max((len(run.group()) for run in cls._white.finditer(rest)), default=0)
        longest = max(longest, 2)
        return count * len(rest) * longest * longest > cls._SCAN_BUDGET

    def __init__(self, tools: list[dict[str, Any]] | None, *, request: dict | None = None):
        self._stream_cut: int | None = None
        self._stream_scanned = 0      # no opener starts before this offset of the cumulative answer
        self._stream_seen = ""        # the answer of the last stream() call
        self.tools = tools or []
        self.declared_tools = (request.get("tools") or []) if request is not None else self.tools
        self.tools_field = ("omitted" if request is not None and "tools" not in request else
                            "nonempty" if self.declared_tools else "empty")
        choice = request.get("tool_choice", "auto") if request is not None else "auto"
        if isinstance(choice, dict):
            function = choice.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            self.tool_choice = {"type": "function", "name": str(name or "")[:96]}
        else:
            self.tool_choice = str(choice)[:96]

    _openers = ("<tool_call>", "<｜DSML｜ calls>")

    def stream(self, text: str) -> str:
        """Release cumulative prose until a candidate, then validate at the end.

        Only a possible opening tag suffix waits for more characters. Once an
        executable opener occurs, the rest waits for final atomic validation.
        No call delta can escape before a later unoffered call is checked. Every opener waits, wherever it is
        (the guard reads no Markdown): only parse() sees the whole call.
        """
        if not text.startswith(self._stream_seen):
            # The text does not extend the text of the last call (a reset or a rewrite): trust nothing, scan again.
            self._stream_cut, self._stream_scanned = None, 0
        self._stream_seen = text
        if self._stream_cut is not None:
            return text[:self._stream_cut]
        # Normally the answer only grows, so each call scans the new text plus the longest opener prefix again.
        longest = max(len(tag) for tag in self._openers)
        first = min(self._stream_scanned, len(text))
        for opening in self._opener_re.finditer(text, first):
            self._stream_cut = opening.start()
            return text[:self._stream_cut]
        tail = text[-(longest - 1):]
        held = 0
        for tag in self._openers:                      # (the call tag in any case, the DSML tag as the parser reads it)
            exact = "DSML" in tag
            for size in range(1, len(tag)):
                if (tail if exact else tail.lower()).endswith(tag[:size] if exact else tag[:size].lower()):
                    held = max(held, size)
        self._stream_scanned = max(first, len(text) - (longest - 1))
        return text[:-held] if held else text

    def parse(self, text: str, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
        """Return CUDA's parsed result, or fail atomically on a call that is not offered.

        The guard reads calls with the engine's own parser rules (``tensorfold.cuda.reply_text``): it rewrites DSML
        blocks and finds call blocks with the engine's own expressions, reads the name of each block as the engine
        does (also its single-call checks when ``max_calls`` is set), and rejects the answer when a name is not
        offered. The result is the engine's own ``parse_tool_calls`` on the same text, so the guard and the parser
        cannot disagree about what is a call.
        """

        from tensorfold.cuda import reply_text
        from tensorfold.server.tools import tool_spec_name

        def reject(name: str, reason: str, start: int, stop: int, kind: str) -> ModelOutputError:
            effective = sorted({tool_spec_name(tool) for tool in self.tools if tool_spec_name(tool)})
            declared = sorted({tool_spec_name(tool) for tool in self.declared_tools if tool_spec_name(tool)})
            return ModelOutputError(name, diagnostic={
                "reason": reason, "wrapper": kind, "span_start": start, "span_end": stop,
                "answer_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "effective_tools": [value[:96] for value in effective[:64]],
                "effective_tool_count": len(effective),
                "declared_tools": [value[:96] for value in declared[:64]],
                "declared_tool_count": len(declared),
                "tools_field": self.tools_field, "tool_choice": self.tool_choice,
            })

        known = {reply_text._tool_name(tool).lower() for tool in self.tools if reply_text._tool_name(tool)}
        # A DSML block ends at an exact closer, so text after the last closer holds no block. The rewrite reads only
        # the text up to that closer: the regex would try every unclosed opener against the rest of the answer.
        close = text.rfind(reply_text._DSML_CLOSE)
        end = close + len(reply_text._DSML_CLOSE)
        head, tail = (text[:end], text[end:]) if close >= 0 else ("", text)
        dsml_tag = reply_text._DSML
        for dsml in reply_text._DSML_BLOCK_RE.finditer(head):          # (inside each block, then each invoke)
            if self._stalls(dsml.group(1), f'<{dsml_tag} invoke name="', f"</{dsml_tag} invoke>"):
                raise reject("", "runaway_markup", dsml.start(), dsml.end(), "dsml")
            for invoke in reply_text._DSML_INVOKE_RE.finditer(dsml.group(1)):
                if self._stalls(invoke.group(2), f'<{dsml_tag} parameter name="', f"</{dsml_tag} parameter>"):
                    raise reject("", "runaway_markup", dsml.start(), dsml.end(), "dsml")
        try:
            converted = reply_text.dsml_to_tool_calls(head) + tail
        except (ValueError, RecursionError):           # a value that the JSON decoder refuses (for example 5000 digits)
            raise reject("", "undecodable_call", 0, len(text), "dsml") from None
        lowered = self._fold(converted)
        if reply_text._CALL_CLOSE not in lowered:
            # No call block can exist. (The parser would try every opener against the rest of the answer.)
            return (converted.strip(), None) if self.tools else (text, None)
        if self._stalls(lowered, reply_text._CALL_OPEN, reply_text._CALL_CLOSE):
            raise reject("", "runaway_markup", 0, len(text), "json")
        # Each rewritten block: its span in the converted text and its DSML span in the answer (for the diagnostic).
        segments: list[tuple[int, int, int, int]] = []
        shift = 0
        for dsml in reply_text._DSML_BLOCK_RE.finditer(head):
            made = reply_text.dsml_to_tool_calls(dsml.group(0))
            segments.append((dsml.start() + shift, dsml.start() + shift + len(made), dsml.start(), dsml.end()))
            shift += len(made) - (dsml.end() - dsml.start())
        starts = [segment[0] for segment in segments]
        stops = [segment[1] for segment in segments]

        def source_span(first: int, last: int) -> tuple[int, int, bool]:
            """The span of a block of the converted text in the answer, and whether a DSML rewrite made it."""

            inside = bisect.bisect_right(starts, first) - 1
            if inside >= 0 and first < segments[inside][1]:
                return segments[inside][2], segments[inside][3], True
            def moved(position: int) -> int:             # the answer is longer or shorter than the converted text
                before = bisect.bisect_right(stops, position) - 1
                return position + (segments[before][3] - segments[before][1] if before >= 0 else 0)
            end = bisect.bisect_right(starts, last - 1) - 1
            return moved(first), (segments[end][3] if end >= 0 and last - 1 < segments[end][1] else moved(last)), False

        # The parser's single-call checks (leftover text after the arguments) matter only when it can make a call, that
        # is with offered tools. Without tools the guard reads names only, as in the default mode (fail-closed).
        single = max_calls is not None and bool(self.tools)
        blocks = list(reply_text._TOOL_CALL_BLOCK_RE.finditer(converted))
        if not blocks:                                 # (the parser's result for an answer without a call block)
            return (converted.strip(), None) if self.tools else (text, None)
        for block in blocks:
            start, stop, made = source_span(block.start(), block.end())
            wrapper = "dsml" if made else "json"
            try:
                names = self._call_names(reply_text, block.group(1).strip(), complete=single)
            except _RunawayMarkup:                     # (only where the parser would try the Qwen and GLM forms)
                raise reject("", "runaway_markup", start, stop, wrapper) from None
            except (ValueError, RecursionError):       # a value that the JSON decoder refuses
                raise reject("", "undecodable_call", start, stop, wrapper) from None
            for name, kind in names:
                if name.lower() not in known:
                    declared = {tool_spec_name(tool).lower() for tool in self.declared_tools if tool_spec_name(tool)}
                    raise reject(name, "disallowed_by_tool_choice" if name.lower() in declared else "unavailable_tool",
                                 start, stop, "dsml" if wrapper == "dsml" else kind)
        try:
            return reply_text.parse_tool_calls(text, self.tools, max_calls=max_calls)
        except (ValueError, RecursionError):           # arguments that the parser cannot decode
            raise reject("", "undecodable_call", 0, len(text), "json") from None

    def _call_names(self, reply_text: Any, block: str, complete: bool = False) -> list[tuple[str, str]]:
        """The (tool name, wrapper) that the engine's parser may read from one call block (JSON, Qwen XML or GLM).
        ``complete`` is the parser's single-call mode (``max_calls`` set): it refuses Qwen and GLM bodies with extra text."""

        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            return self._text_names(reply_text, block, complete)   # not JSON: the parser tries the Qwen and GLM forms
        if not isinstance(payload, dict):
            return []                                  # (valid JSON that is not an object is no call for the parser)
        function = payload.get("function")
        for source in (function, payload):             # (an empty nested object falls back to the flat form)
            name = source.get("name") if isinstance(source, dict) else None
            if name and str(name).strip():
                break
        else:
            return []
        # The parser matches the name as written. Offered names are trimmed, so a name with outer white space can
        # never match one: it is no call.
        if str(name) != str(name).strip():
            return []
        arguments = source.get("arguments", source.get("parameters", {}))
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:                   # the parser then tries the Qwen and GLM forms, and keeps the name
            others = self._text_names(reply_text, block, complete)
            return others if others else ([] if complete else [(str(name), "json")])
        if complete:                                   # the single-call mode accepts a finite object only
            try:
                if not isinstance(arguments, dict):
                    return []
                json.dumps(arguments, allow_nan=False)
            except (ValueError, TypeError):
                return []
        return [(str(name), "json")]

    def _fallback_stalls(self, block: str, form: str | None = None) -> bool:
        """Whether the Qwen or GLM expressions of the parser would scan this call block for long. The parser reads these
        forms only when the block is not a JSON call, so this is checked only then (a JSON argument may hold any text)."""

        qwen = self._fold(block)                       # (the Qwen expressions ignore case; the GLM tags are literal)
        return ((form != "glm" and any(self._stalls(qwen, opener, closer)
                    for opener, closer in (("<function=", "</function>"), ("<parameter=", "</parameter>"))))
                or (form != "qwen_xml" and (any(self._stalls(block, opener, closer)
                       for opener, closer in (("<arg_key>", "</arg_key>"), ("<arg_value>", "</arg_value>"),
                                              ("<arg_key>", "<arg_value>")))   # (a key needs a value to make a pair)
                # A value without its closer sends the pairing expression back over every key before it.
                or (self._open_tail(block, "<arg_value>", "</arg_value>") > 0
                    and block.count("<arg_key>") > self._RUNAWAY))))

    def _text_names(self, reply_text: Any, block: str, complete: bool) -> list[tuple[str, str]]:
        """The (tool name, wrapper) of a Qwen XML or GLM call block."""

        from tensorfold.server import tools as engine_tools

        # Select the parser's form before checking its tags. Other-form tags can be literal argument text.
        head = self._qwen_head.match(block)
        qwen = head is not None and self._qwen_close.search(block, head.end()) is not None
        name_re = getattr(engine_tools, "_GLM_NAME_RE", None)
        name = block.partition("<arg_key>")[0].strip()
        form = "qwen_xml" if qwen else "glm" if name_re is not None and name_re.fullmatch(name) else None
        if self._fallback_stalls(block, form):
            raise _RunawayMarkup
        if not complete:
            # The Qwen name and the closing tag, read linearly. (The parser expression has white space on both sides of
            # a lazy part: on a body without its closing tag it backtracks for long. The closer is matched with the
            # same case rules as the parser.)
            if qwen:
                return [(head.group(1).strip(), "qwen_xml")]
        else:
            xml = reply_text._TOOL_FUNCTION_BLOCK_RE.match(block)
            if xml:
                if reply_text._TOOL_PARAMETER_BLOCK_RE.sub("", xml.group(2)).strip():
                    return []
                return [(xml.group(1).strip(), "qwen_xml")]
        if not complete:
            # The name of a GLM call is the text before its first key. The parser's pairing expression for the keys
            # and values is not needed for the name, and it can be quadratic on malformed output.
            if name_re is not None:
                return [(name, "glm")] if name_re.fullmatch(name) else []
        glm = reply_text.parse_glm_tool_call_block(block, self.tools, complete=complete)
        return [(str(glm[0]).strip(), "glm")] if glm is not None and str(glm[0]).strip() else []
