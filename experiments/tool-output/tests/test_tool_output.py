import json
import unittest
from unittest.mock import patch


from support import fixtures, guard_module


def load():
    return guard_module()


class ToolOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load()
        cls.tools = [{"type": "function", "function": {"name": "task", "parameters": {"type": "object"}}}]

    def setUp(self):
        self.fixture_context = fixtures()
        self.fixture_context.__enter__()
        self.addCleanup(self.fixture_context.__exit__, None, None, None)

    def guard(self):
        return self.module.ToolOutputGuard(self.tools)

    def test_prose_streams_for_absent_empty_and_offered_tools(self):
        for body in ({}, {"tools": []}, {"tools": self.tools}):
            guard = self.module.ToolOutputGuard(body.get("tools"), request=body)
            text = "Ordinary streamed prose."
            for size in range(1, len(text) + 1):
                self.assertEqual(guard.stream(text[:size]), text[:size])
            self.assertEqual(guard.parse(text), (text, None))

    def test_split_openers_hold_only_candidate_not_prose(self):
        for opener in ("<tool_call>", "<TOOL_CALL>", "<｜DSML｜ calls>"):
            for split in range(1, len(opener)):
                guard = self.guard()
                self.assertEqual(guard.stream("prefix " + opener[:split]), "prefix ")
                self.assertEqual(guard.stream("prefix " + opener + "arguments"), "prefix ")
                self.assertEqual(guard.stream("prefix " + opener + "arguments and more"), "prefix ")
        guard = self.guard()
        self.assertEqual(guard.stream("a <too"), "a ")
        self.assertEqual(guard.stream("a <tooth"), "a <tooth")

    def test_real_argument_backticks_do_not_hide_later_calls(self):
        for argument in ("`", "```\n", "~~~\n", '"', "``a ` b"):
            first = '<tool_call>' + json.dumps({"name": "task", "arguments": {"note": argument}}) + '</tool_call>'
            second = '<tool_call>{"name":"task","arguments":{"second":true}}</tool_call>'
            content, calls = self.guard().parse(first + '\n' + second)
            self.assertEqual(content, '')
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"note": argument})
            with self.assertRaises(self.module.ModelOutputError):
                self.guard().parse(first + '\n' + second.replace('"task"', '"unoffered"'))

    def test_unknown_call_rejected_with_no_tools(self):
        for body in ({}, {"tools": []}):
            guard = self.module.ToolOutputGuard(body.get("tools"), request=body)
            with self.assertRaises(self.module.ModelOutputError):
                guard.parse('<tool_call>{"name":"task","arguments":{}}</tool_call>')

    def assert_unavailable(self, text):
        with self.assertRaises(self.module.ModelOutputError) as raised:
            self.guard().parse(text)
        self.assertEqual(raised.exception.code, "unavailable_tool_call")
        self.assertEqual(raised.exception.error["type"], "model_output_error")
        self.assertEqual(raised.exception.error["code"], "unavailable_tool_call")
        self.assertEqual(raised.exception.error["source"], "tensorfold.tool_output_guard")
        self.assertIn(raised.exception.rejection_id, raised.exception.error["message"])
        self.assertNotIn("todowrite", str(raised.exception.error))
        self.assertNotIn("secret", str(raised.exception.error))

    def test_diagnostics_identify_wrapper_and_tools_without_arguments(self):
        cases = (
            ('<tool_call>{"name":"unoffered","arguments":{"token":"PRIVATE_PAYLOAD"}}</tool_call>', "json"),
            ('<tool_call><function=unoffered><parameter=x>PRIVATE_PAYLOAD</parameter></function></tool_call>', "qwen_xml"),
            ('<tool_call>unoffered<arg_key>x</arg_key><arg_value>PRIVATE_PAYLOAD</arg_value></tool_call>', "glm"),
            ('<｜DSML｜ calls><｜DSML｜ invoke name="unoffered"></｜DSML｜ invoke></｜DSML｜ calls>', "dsml"),
        )
        for text, kind in cases:
            with self.subTest(kind=kind), self.assertRaises(self.module.ModelOutputError) as raised:
                self.guard().parse(text)
            exc = raised.exception
            self.assertEqual(exc.diagnostic["wrapper"], kind)
            self.assertEqual(exc.diagnostic["tool_name"], "unoffered")
            self.assertEqual(exc.diagnostic["effective_tools"], ["task"])
            self.assertEqual(exc.diagnostic["span_start"], 0)
            self.assertEqual(exc.diagnostic["span_end"], len(text))
            self.assertEqual(len(exc.diagnostic["answer_sha256"]), 64)
            self.assertNotIn("PRIVATE_PAYLOAD", exc.log_line())
            self.assertEqual(json.loads(exc.log_line().split("unavailable_tool_call ", 1)[1]), exc.diagnostic)

    def test_filtered_tools_are_distinct_from_unoffered_tools_in_diagnostics(self):
        guard = self.module.ToolOutputGuard([], request={"tools": self.tools, "tool_choice": "none"})
        with self.assertRaises(self.module.ModelOutputError) as raised:
            guard.parse('<tool_call>{"name":"task","arguments":{}}</tool_call>')
        info = raised.exception.diagnostic
        self.assertEqual(info["reason"], "disallowed_by_tool_choice")
        self.assertEqual(info["declared_tools"], ["task"])
        self.assertEqual(info["effective_tools"], [])
        self.assertEqual(info["tool_choice"], "none")

    def test_diagnostics_escape_log_control_characters_and_bound_names(self):
        name = "unavailable\n" + "x" * 300
        text = '<tool_call>' + json.dumps({"name": name, "arguments": {"secret": "PRIVATE_PAYLOAD"}}) + '</tool_call>'
        with self.assertRaises(self.module.ModelOutputError) as raised:
            self.guard().parse(text)
        self.assertEqual(len(raised.exception.diagnostic["tool_name"]), 96)
        self.assertNotIn("\n", raised.exception.log_line())
        self.assertNotIn("PRIVATE_PAYLOAD", raised.exception.log_line())

    def test_benign_json_task_payload_is_preserved_by_cuda_parser(self):
        content, calls = self.guard().parse('before <tool_call>{"name":"task","arguments":{"saved":"benign"}}</tool_call> after')
        self.assertEqual(content, "before  after")
        self.assertEqual(calls[0]["function"]["name"], "task")
        self.assertEqual(calls[0]["function"]["arguments"], '{"saved":"benign"}')

    def test_rejects_unknown_json_xml_glm_and_dsml(self):
        for text in (
            '<tool_call>{"function":{"name":"todowrite","arguments":{"secret":1}}}</tool_call>',
            '<tool_call><function=todowrite><parameter=x>secret</parameter></function></tool_call>',
            '<tool_call>todowrite<arg_key>x</arg_key><arg_value>secret</arg_value></tool_call>',
            '<｜DSML｜ calls><｜DSML｜ invoke name="todowrite"></｜DSML｜ invoke></｜DSML｜ calls>',
        ):
            with self.subTest(text=text):
                self.assert_unavailable(text)

    def test_unknown_after_valid_call_fails_even_at_limit(self):
        text = '<tool_call>{"name":"task","arguments":{}}</tool_call><tool_call>{"name":"todowrite","arguments":{}}</tool_call>'
        with self.assertRaises(self.module.ModelOutputError):
            self.guard().parse(text, max_calls=1)

    def test_reported_task_envelope_is_rejected_when_task_is_not_offered(self):
        tools = [{"type": "function", "function": {"name": "todowrite", "parameters": {"type": "object"}}}]
        text = '<tool_call>{"name":"task","arguments":{"description":"Trace spill internals","prompt":"Research only, no edits.","subagent_type":"explore","task_id":""}}</tool_call>'
        with self.assertRaises(self.module.ModelOutputError):
            self.module.ToolOutputGuard(tools).parse(text)

    def test_a_closing_tag_inside_a_json_string_ends_the_block_for_the_guard_and_the_parser_alike(self):
        text = '<tool_call>{"name":"todowrite","arguments":{"secret":"literal </tool_call> text"}}</tool_call>'
        from tensorfold.cuda import reply_text
        self.assertEqual(self.guard().parse(text), reply_text.parse_tool_calls(text, self.tools))
        self.assertIsNone(self.guard().parse(text)[1])                            # not a call for the parser: none runs

    def test_argument_code_and_quotes_are_not_rewritten(self):
        import json
        args = {"prompt": 'Explain `x` and ``a ` b``. Quote: "example".\n```python\nx = 1\n```'}
        text = '<tool_call>' + json.dumps({"name": "task", "arguments": args}) + '</tool_call>'
        content, calls = self.guard().parse(text)
        self.assertEqual(content, "")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), args)

    def test_offered_calls_keep_prefix_and_obey_global_limit(self):
        text = 'prefix <tool_call>{"name":"TASK","arguments":{}}</tool_call> middle <tool_call>{"name":"task","arguments":{}}</tool_call> suffix'
        content, calls = self.guard().parse(text, max_calls=1)
        self.assertEqual(content, "prefix  middle  suffix")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "task")

    def test_streaming_never_releases_a_call_that_follows_a_quote_mark(self):
        call = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for text in (f'He said "go"\n{call}', f'"{call}"'):
            with self.subTest(text=text):
                guard = self.guard()
                for size in range(1, len(text) + 1):
                    shown = guard.stream(text[:size])
                    self.assertNotIn("PRIVATE", shown)
                    self.assertNotIn("<tool_call>", shown)

    def test_flat_name_is_used_when_the_nested_function_object_is_empty(self):
        for payload in ('{"name":"unoffered","function":{},"arguments":{"secret":"PRIVATE"}}',
                        '{"name":"unoffered","function":{"name":""},"arguments":{}}',
                        '{"name":"unoffered","function":"x","arguments":{}}'):
            with self.subTest(payload=payload):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(f"<tool_call>{payload}</tool_call>")
        content, calls = self.guard().parse('<tool_call>{"name":"task","function":{},"arguments":{}}</tool_call>')
        self.assertIsNone(calls)                         # the parser reads the empty nested object: no name, no call

    def test_a_call_example_in_a_code_fence_is_a_call(self):
        # The guard reads no Markdown: an offered example in a fence is a call, and so is the call after it.
        literal = '<tool_call>{"name":"task","arguments":{"example":true}}</tool_call>'
        real = '<tool_call>{"name":"task","arguments":{"example":false}}</tool_call>'
        content, calls = self.guard().parse('```json\n' + literal + '\n```\n' + real)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("tool_call", content)

    def test_streaming_scans_only_new_text(self):
        positions = []
        real = self.module.ToolOutputGuard._opener_re

        class Spy:
            @staticmethod
            def finditer(text, position=0):
                positions.append(position)
                return real.finditer(text, position)

        guard = self.guard()
        text = "word " * 400
        with patch.object(self.module.ToolOutputGuard, "_opener_re", Spy):
            for size in range(50, len(text) + 1, 50):
                self.assertEqual(guard.stream(text[:size]), text[:size])
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(positions[-1], len(text) - 100)               # the last call skipped the old text

    def test_incremental_streaming_gives_the_same_result_as_a_fresh_scan(self):
        literal = '<tool_call>{"name":"task","arguments":{"note":"`"}}</tool_call>'
        real = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for text in (f"'{literal}' and then {real}", f"```\n{literal}\n```\nmore {real}", f'say "{literal}" now'):
            with self.subTest(text=text):
                guard = self.guard()
                for size in range(1, len(text) + 1):
                    self.assertEqual(guard.stream(text[:size]), self.guard().stream(text[:size]))

    def test_stream_scans_again_when_the_text_does_not_extend_the_last_text(self):
        call = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for first, second in (("x" * 60, call + "x" * 60),            # rewritten, same length or longer
                              ("x" * 60, call[:20]),                   # shorter than the last text
                              ("y" * 40 + " ", call + " tail")):       # a new buffer with a call before the old end
            with self.subTest(second=second[:30]):
                guard = self.guard()
                guard.stream(first)
                shown = guard.stream(second)
                self.assertNotIn("PRIVATE", shown)
                self.assertNotIn("<tool_call>", shown)
        guard = self.guard()                                           # also after the cut was set
        guard.stream("ok " + call)
        self.assertEqual(guard.stream("fresh text"), "fresh text")
        self.assertNotIn("<tool_call>", guard.stream(call + " more"))

    def test_malformed_and_truncated_wrappers_are_not_misclassified(self):
        for text in (
            '<tool_call>{"name":}</tool_call>',
            '<tool_call>{"name":"todowrite","arguments":{}',
            'mention <tool_call> only',
        ):
            with self.subTest(text=text):
                content, calls = self.guard().parse(text)
                self.assertIsNone(calls)
                self.assertIn(text.replace("</tool_call>", "") if text.endswith("{}") else text[:10], content)

    def test_a_call_inside_a_closed_code_fence_is_rejected_like_any_other_call(self):
        # Known limit of a guard that fails closed: an example of an unoffered call in a code fence is rejected.
        call = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for text in (f"```\n{call}\n```\n", f"```json\n{call}\n```", f"~~~\n{call}\n~~~\n", f"````\n{call}\n````\n",
                     f"   ```\n{call}\n   ```\n", f"```\r\n{call}\r\n```\r\n", f"intro\n\n```\n{call}\n```\nafter",
                     f"```\n{call}\n\n{call}\n```\n"):
            with self.subTest(text=text):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(text)

    def test_every_other_case_is_a_real_call_and_is_rejected(self):
        # Quotes, quote lines, inline code, indented code, escapes and fences that are not sure all fail closed.
        call = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        cases = {
            "double quotes": f'"{call}"', "single quotes": f"'{call}'", "spaced quotes": f'"  {call}  "',
            "inline code": f"`{call}`", "long inline code": f"``{call}``", "quote line": f"> {call}",
            "nested quote line": f">> {call}", "unclosed fence": f"```\n{call}\n", "fence closed before": f"```\nx\n```\n{call}",
            "fence opened after": f"{call}\n```\nx\n```", "four spaces of indent": f"    ```\n{call}\n    ```",
            "indented code": f"    {call}", "tab indent": f"\t```\n{call}\n\t```",
            "closer of another kind": f"```\n{call}\n~~~", "closer too short": f"````\n{call}\n```",
            "closer with text": f"```\n{call}\n``` text", "backtick in the info string": f"```lang```x\n{call}\n```",
            "quote across lines": f'He said "go\nnow" {call} "and left"',
            "adjacent quotes": f'The user said "go" {call} "and left."', "escaped quote": f'Example: \\" {call} "tail',
            "escaped backtick": f"Example: \\` {call} `", "quote marks in code": f'He said "the mark `"` " {call} "and next"',
            "blank line in a quote": f"> \n\n{call}", "CRLF blank line in a quote": f"> \r\n\r\n{call}",
            "span closed inside": f"`a\\` {call}",
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(text)

    def test_offered_calls_in_every_other_case_run_as_calls(self):
        call = '<tool_call>{"name":"task","arguments":{}}</tool_call>'
        for text in (f'"{call}"', f"`{call}`", f"> {call}", f"```\n{call}\n", f"    ```\n{call}\n    ```"):
            with self.subTest(text=text):
                content, calls = self.guard().parse(text)
                self.assertEqual(len(calls), 1)

    def test_fences_inside_call_arguments_cannot_make_a_later_call_literal(self):
        offered = "<tool_call><function=task><parameter=code>\n```python\nx = 1\n</parameter></function></tool_call>"
        closing = "<tool_call><function=task><parameter=code>\n```\n</parameter></function></tool_call>"
        real = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        with self.assertRaises(self.module.ModelOutputError):
            self.guard().parse(offered + "\n" + real + "\n" + closing)

    def test_a_call_after_a_fence_mark_does_not_turn_the_line_into_a_closing_line(self):
        unoffered = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        offered = '<tool_call>{"name":"task","arguments":{}}</tool_call>'
        for text in (f"```\n{unoffered}\n``` {offered}", f"```\n{unoffered}\n```{offered}\n",
                     f"~~~\n{unoffered}\n~~~ {offered}"):
            with self.subTest(text=text):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(text)

    def test_a_looping_model_with_many_unclosed_openers_is_read_fast(self):
        import time

        text = "<tool_call>{" * 20000
        self.assertEqual(self.guard().parse(text), (text, None))                  # no closing tag: no call block
        start = time.perf_counter()
        content, calls = self.guard().parse(text + "</tool_call>")                # one block: not a call (broken JSON)
        self.assertIsNone(calls)
        self.assertLess(time.perf_counter() - start, 5.0)

    def test_every_dsml_name_the_parser_reads_is_checked(self):
        for name in ("$TOOL_NAME", "mcp/tool", "a b", "tool{x}"):
            dsml = f'<｜DSML｜ calls><｜DSML｜ invoke name="{name}"> </｜DSML｜ invoke></｜DSML｜ calls>'
            with self.subTest(name=name):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(dsml)
        offered = [{"type": "function", "function": {"name": "mcp/tool", "parameters": {"type": "object"}}}]
        dsml = '<｜DSML｜ calls><｜DSML｜ invoke name="mcp/tool"> </｜DSML｜ invoke></｜DSML｜ calls>'
        self.module.ToolOutputGuard(offered).parse(dsml)             # an offered name is not an error

    def test_every_call_the_parser_finds_is_checked_by_the_guard(self):
        # The guard and the parser must agree by construction: whenever the engine's parser finds a call named N
        # (N offered), the guard must reject the same text when N is not offered.
        from tensorfold.cuda import reply_text
        from support import parity_samples

        found = 0
        for name, text in parity_samples():
            offered = [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}]
            _, calls = reply_text.parse_tool_calls(text, offered)
            if not calls:
                continue
            found += 1
            with self.subTest(name=name, text=text[:60]):
                with self.assertRaises(self.module.ModelOutputError):
                    self.module.ToolOutputGuard([{"type": "function", "function": {"name": "other"}}]).parse(text)
                _, got = self.module.ToolOutputGuard(offered).parse(text)           # offered: the parser's own result
                self.assertEqual([call["function"]["name"] for call in got], [call["function"]["name"] for call in calls])
        self.assertGreater(found, 30)

    def test_valid_json_that_is_not_an_object_is_no_call_for_the_guard_as_for_the_parser(self):
        from tensorfold.cuda import reply_text

        for body in ("123", '"todowrite"', "[1]", "true", "null", "1.5"):
            text = f"<tool_call>{body}</tool_call>"
            with self.subTest(body=body):
                self.assertEqual(self.guard().parse(text), reply_text.parse_tool_calls(text, self.tools))
                self.assertIsNone(self.guard().parse(text)[1])

    def test_a_name_with_outer_white_space_is_no_call_for_the_guard_as_for_the_parser(self):
        from tensorfold.cuda import reply_text

        for text in ('<tool_call>{"name":" unoffered ","arguments":{}}</tool_call>',
                     '<｜DSML｜ calls><｜DSML｜ invoke name=" unoffered "> </｜DSML｜ invoke></｜DSML｜ calls>'):
            with self.subTest(text=text[:40]):
                offered = [{"type": "function", "function": {"name": " unoffered ", "parameters": {"type": "object"}}}]
                self.assertIsNone(reply_text.parse_tool_calls(text, offered)[1])      # trimmed spec name: never matches
                self.assertEqual(self.guard().parse(text), reply_text.parse_tool_calls(text, self.tools))

    def test_a_value_the_json_decoder_refuses_is_a_safe_rejection(self):
        text = '<tool_call>{"name":"task","arguments":{"n":' + "9" * 6000 + "}}</tool_call>"
        for guard in (self.guard(), self.module.ToolOutputGuard(None), self.module.ToolOutputGuard([])):
            with self.assertRaises(self.module.ModelOutputError) as caught:
                guard.parse(text)
            self.assertEqual(caught.exception.diagnostic["reason"], "undecodable_call")
            self.assertNotIn("9999", caught.exception.log_line())

    def test_a_dsml_value_the_json_decoder_refuses_is_a_safe_rejection(self):
        text = ('<｜DSML｜ calls><｜DSML｜ invoke name="task"><｜DSML｜ parameter name="n" string="false">'
                + "9" * 6000 + "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>")
        for guard in (self.guard(), self.module.ToolOutputGuard(None)):
            with self.assertRaises(self.module.ModelOutputError) as caught:
                guard.parse(text)
            self.assertEqual(caught.exception.diagnostic["reason"], "undecodable_call")

    def test_single_call_mode_reads_names_as_the_parser_does(self):
        from tensorfold.cuda import reply_text

        for text in ("<tool_call>unoffered<arg_key>k</arg_key><arg_value>v</arg_value>xvjunk</tool_call>",
                     "<tool_call>unoffered<arg_key>k</arg_key><arg_value>v</arg_value> junk</tool_call>",
                     "<tool_call><function=unoffered><parameter=k>v</parameter> junk</function></tool_call>"):
            with self.subTest(text=text):
                self.assertEqual(self.guard().parse(text, max_calls=1),
                                 reply_text.parse_tool_calls(text, self.tools, max_calls=1))
                self.assertIsNone(self.guard().parse(text, max_calls=1)[1])       # malformed in this mode: no call
                with self.assertRaises(self.module.ModelOutputError):             # the default mode rejects the name
                    self.guard().parse(text)

    def test_nested_arguments_that_exhaust_the_decoder_are_a_safe_rejection(self):
        deep = "[" * 100000 + "]" * 100000
        json_call = '<tool_call>{"name":"task","arguments":{"x":' + deep + '}}</tool_call>'
        dsml_call = ('<｜DSML｜ calls><｜DSML｜ invoke name="task"><｜DSML｜ parameter name="x" string="false">'
                     + deep + "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>")
        for text in (json_call, dsml_call):
            for guard in (self.guard(), self.module.ToolOutputGuard(None)):
                with self.subTest(text=text[:40], tools=bool(guard.tools)):
                    with self.assertRaises(self.module.ModelOutputError) as caught:
                        guard.parse(text)
                    self.assertEqual(caught.exception.diagnostic["reason"], "undecodable_call")

    def test_a_value_that_only_the_final_parser_refuses_is_a_safe_rejection(self):
        # No usable name: the guard does not decode the arguments, and the parser's own decoder refuses the number.
        text = '<tool_call>{"arguments":"' + "9" * 6000 + '"}</tool_call>'
        with self.assertRaises(self.module.ModelOutputError) as caught:
            self.guard().parse(text)
        self.assertEqual(caught.exception.diagnostic["reason"], "undecodable_call")

    def test_json_arguments_are_checked_as_the_parser_checks_them(self):
        from tensorfold.cuda import reply_text

        for arguments in ('"{"', "[1]", '"[1]"', '{"a":NaN}', '"{\\"a\\":NaN}"', "5"):
            text = '<tool_call>{"name":"unoffered","arguments":' + arguments + '}</tool_call>'
            with self.subTest(arguments=arguments):
                # Single-call mode: the parser finds no call, so the guard does not reject the name.
                self.assertEqual(self.guard().parse(text, max_calls=1),
                                 reply_text.parse_tool_calls(text, self.tools, max_calls=1))
                self.assertIsNone(self.guard().parse(text, max_calls=1)[1])
                with self.assertRaises(self.module.ModelOutputError):         # the default mode rejects the name
                    self.guard().parse(text)

    def test_many_unclosed_dsml_openers_are_read_fast(self):
        import time

        for text in ("<｜DSML｜ calls> " * 30000, "<｜DSML｜ calls> " * 30000 + "</tool_call>"):
            for guard in (self.guard(), self.module.ToolOutputGuard(None)):
                start = time.perf_counter()
                guard.parse(text)
                self.assertLess(time.perf_counter() - start, 5.0)

    def test_the_rejected_span_points_into_the_answer_after_a_dsml_rewrite(self):
        offered = ('<｜DSML｜ calls><｜DSML｜ invoke name="task"><｜DSML｜ parameter name="n" string="false">1'
                   '</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>')
        plain = '<tool_call>{"name":"unoffered","arguments":{}}</tool_call>'
        dsml = '<｜DSML｜ calls><｜DSML｜ invoke name="unoffered"></｜DSML｜ invoke></｜DSML｜ calls>'
        for text, wrapper, wanted in ((offered + " middle " + plain, "json", plain),
                                      (plain.replace("unoffered", "task") + offered + dsml, "dsml", dsml),
                                      (offered + offered + "x" + plain + offered, "json", plain)):
            with self.subTest(wrapper=wrapper, text=text[-40:]):
                with self.assertRaises(self.module.ModelOutputError) as caught:
                    self.guard().parse(text)
                diagnostic = caught.exception.diagnostic
                self.assertEqual(text[diagnostic["span_start"]:diagnostic["span_end"]], wanted)
                self.assertEqual(diagnostic["wrapper"], wrapper)

    def test_a_looping_model_with_open_markup_is_rejected_before_the_parser_scans_it(self):
        import time

        call = '<tool_call>{"name":"task","arguments":{}}</tool_call>'
        invoke = '<｜DSML｜ invoke name="a">'
        cases = (call + "<tool_call>{" * 30000,                                   # openers after the last closer
                 "<｜DSML｜ calls>" + invoke * 30000 + "</｜DSML｜ calls>" + call,    # open invokes in a DSML block
                 "<tool_call><function=task>" + "<parameter=x>" * 30000 + "</function></tool_call>",
                 "<tool_call>task" + "<arg_key>k" * 30000 + "</tool_call>",
                 # as many closers as openers, but the openers come last: no closer after them
                 "<｜DSML｜ calls>" + "</｜DSML｜ invoke>" * 30000 + invoke * 30000 + "</｜DSML｜ calls>" + call,
                 "<tool_call><function=task>" + "</parameter>" * 30000 + "<parameter=x>" * 30000 + "</function></tool_call>",
                 # one block with open tags, a later block that is balanced
                 "<｜DSML｜ calls>" + invoke * 30000 + "</｜DSML｜ calls><｜DSML｜ calls>" + invoke + "</｜DSML｜ invoke></｜DSML｜ calls>",
                 "<tool_call>task" + "<arg_value>k" * 30000 + "</tool_call><tool_call>task<arg_value>k</arg_value></tool_call>",
                 "<tool_call>task" + "<arg_key>k</arg_key>" * 30000 + "</tool_call>",             # keys without values
                 "<tool_call>task" + "<arg_key>k</arg_key><arg_value>v" * 30000 + "</tool_call>",
                 "<tool_call>task" + "<arg_key>k</arg_key>" * 30000 + "<arg_value>v</tool_call>")        # one open value
        for text in cases:
            for guard in (self.guard(), self.module.ToolOutputGuard(None)):
                with self.subTest(text=text[:40], tools=bool(guard.tools)):
                    start = time.perf_counter()
                    with self.assertRaises(self.module.ModelOutputError) as caught:
                        guard.parse(text)
                    self.assertEqual(caught.exception.diagnostic["reason"], "runaway_markup")
                    self.assertLess(time.perf_counter() - start, 5.0)
        content, calls = self.guard().parse(call + "<tool_call>{" * 60)           # a few open tags are no runaway
        self.assertEqual(len(calls), 1)

    def test_malformed_qwen_and_glm_blocks_are_read_fast_without_tools(self):
        import time

        qwen = "<tool_call><function=task>" + "<parameter=x" * 30000 + ">v</parameter></function></tool_call>"
        glm = "<tool_call>task" + "<arg_key>k</arg_key>" * 30000 + "<arg_value>v</arg_value></tool_call>"
        for text in (qwen, glm):
            for max_calls in (None, 1):
                with self.subTest(text=text[:30], max_calls=max_calls):
                    start = time.perf_counter()
                    with self.assertRaises(self.module.ModelOutputError) as caught:
                        self.module.ToolOutputGuard(None).parse(text, max_calls=max_calls)
                    self.assertEqual(caught.exception.diagnostic["reason"], "unavailable_tool")
                    self.assertLess(time.perf_counter() - start, 5.0)

    def test_an_unterminated_opener_with_a_long_white_space_run_is_not_scanned_for_long(self):
        import time

        invoke = '<｜DSML｜ calls><｜DSML｜ invoke name="a">' + " " * 3000 + "X</｜DSML｜ calls>"
        qwen = "<tool_call><function=task>" + " " * 3000 + "X</tool_call>"
        for text in (invoke, qwen):
            for guard in (self.guard(), self.module.ToolOutputGuard(None)):
                with self.subTest(text=text[:30], tools=bool(guard.tools)):
                    start = time.perf_counter()
                    try:
                        guard.parse(text)
                    except self.module.ModelOutputError:
                        pass
                    self.assertLess(time.perf_counter() - start, 2.0)

    def test_repeated_white_space_runs_after_an_unterminated_opener_are_not_scanned_for_long(self):
        import time

        unit = " " * 256 + "X"
        done = "<tool_call>{}</tool_call>"
        shapes = (done + "<tool_call>" + unit * 64,
                  done + "<tool_call><function=task>" + unit * 64,
                  done + "<tool_call><function=task>" + (" " * 100 + "<parameter=a>") * 300,
                  '<｜DSML｜ calls><｜DSML｜ invoke name="a">' + unit * 64 + "</｜DSML｜ calls>" + "<tool_call>" + unit * 64,
                  done + ("<tool_call>" + "a" * 47000) * 64)             # (no white space: 64 openers, 3 million characters)
        for text in shapes:
            for guard in (self.guard(), self.module.ToolOutputGuard(None), self.module.ToolOutputGuard([], request={"tools": []})):
                with self.subTest(text=text[:40], tools=bool(guard.tools)):
                    start = time.perf_counter()
                    try:
                        guard.parse(text)
                    except self.module.ModelOutputError:
                        pass
                    self.assertLess(time.perf_counter() - start, 0.8)

    def test_a_short_call_with_indented_text_after_an_unterminated_opener_is_not_rejected(self):
        text = "<tool_call>{}</tool_call>\nNote: <tool_call> is the tag.\n" + "    indented line\n" * 20
        guard = self.module.ToolOutputGuard(None)
        guard.parse(text)

    def test_tags_are_read_with_the_case_rules_of_the_parser(self):
        # re.IGNORECASE also matches the dotless i (U+0131) and the long s (U+017F) with an ASCII letter.
        body = '{"x": 1}'
        for closer in ("</functıon>", "</FUNCTİON>", "</function>"):
            text = "<tool_call><function=secret>" + body + closer + "</tool_call>"
            for guard in (self.guard(), self.module.ToolOutputGuard(None)):
                with self.subTest(closer=closer, tools=bool(guard.tools)):
                    with self.assertRaises(self.module.ModelOutputError) as caught:
                        guard.parse(text)
                    self.assertEqual(caught.exception.diagnostic["reason"], "unavailable_tool")
        looping = "<tool_call>" + "<functıon=a>" * 100 + "</tool_call>"
        with self.assertRaises(self.module.ModelOutputError) as caught:
            self.guard().parse(looping)
        self.assertEqual(caught.exception.diagnostic["reason"], "runaway_markup")

    def test_markup_inside_a_json_argument_is_not_looping_output(self):
        # The parser reads the Qwen and GLM forms only when a block is not a JSON call, so markup in a JSON argument
        # (for example a generated document) is an ordinary argument.
        for noise in ("<arg_key>" * 100, "<parameter=a>" * 100, "<function=a>" * 100, "<arg_value>" * 100):
            payload = json.dumps({"name": "task", "arguments": {"text": noise}})
            as_string = json.dumps({"name": "task", "arguments": json.dumps({"text": noise})})
            for body in (payload, as_string):
                with self.subTest(noise=noise[:12], string=body is as_string):
                    content, calls = self.guard().parse("<tool_call>" + body + "</tool_call>")
                    self.assertEqual([call["function"]["name"] for call in calls], ["task"])

    def test_qwen_values_can_hold_many_literal_glm_tags(self):
        from tensorfold.cuda import reply_text

        for value in ("<arg_key>" * 65, "<arg_value>" * 65,
                      "<arg_key>x</arg_key>" * 65 + "<arg_value>unfinished"):
            text = ("<tool_call><function=task><parameter=note>" + value
                    + "</parameter></function></tool_call>")
            for max_calls in (None, 1):
                with self.subTest(value=value[:30], max_calls=max_calls):
                    expected = reply_text.parse_tool_calls(text, self.tools, max_calls=max_calls)
                    content, calls = self.guard().parse(text, max_calls=max_calls)
                    self.assertEqual(content, expected[0])
                    self.assertEqual([call["function"] for call in calls],
                                     [call["function"] for call in expected[1]])
                    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"note": value})

    def test_glm_values_can_hold_many_literal_qwen_tags(self):
        from tensorfold.cuda import reply_text

        for value in ("<function=literal>" * 65, "<parameter=literal>" * 65):
            text = "<tool_call>task<arg_key>note</arg_key><arg_value>" + value + "</arg_value></tool_call>"
            for max_calls in (None, 1):
                with self.subTest(value=value[:30], max_calls=max_calls):
                    expected = reply_text.parse_tool_calls(text, self.tools, max_calls=max_calls)
                    content, calls = self.guard().parse(text, max_calls=max_calls)
                    self.assertEqual(content, expected[0])
                    self.assertEqual([call["function"] for call in calls],
                                     [call["function"] for call in expected[1]])
                    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"note": value})

    def test_upper_case_glm_tags_are_not_glm_markup(self):
        # The GLM tags of the parser are literal and case sensitive: other spellings are text, not a looping call.
        text = "<tool_call>task" + "<ARG_KEY>k</ARG_KEY><ARG_VALUE>v" * 100 + "</tool_call>"
        for guard in (self.guard(), self.module.ToolOutputGuard(None)):
            with self.subTest(tools=bool(guard.tools)):
                try:
                    guard.parse(text)
                except self.module.ModelOutputError as error:
                    self.assertNotEqual(error.diagnostic["reason"], "runaway_markup")

    def test_streaming_holds_the_dsml_opener_only_as_written(self):
        guard = self.guard()
        lower = "long answer " * 5 + "<｜dsml｜ calls> and more prose"
        self.assertEqual(guard.stream(lower), lower)                              # not the parser's tag: plain text
        guard = self.guard()
        upper = "long answer <｜DSML｜ calls> and more"
        self.assertEqual(guard.stream(upper), "long answer ")

    def test_many_envelopes_are_read_in_one_pass(self):
        import time

        call = '<tool_call>{"name":"task","arguments":{"n":1}}</tool_call>'
        text = ("```\n" + call + "\n```\n") * 20000
        start = time.perf_counter()
        content, calls = self.guard().parse(text)
        self.assertEqual(len(calls), 20000)
        self.assertLess(time.perf_counter() - start, 20.0)           # a copy of the answer for each call needs minutes

    def test_a_call_with_a_fence_line_in_its_body_is_rejected(self):
        body = '<function=unoffered><parameter=secret>PRIVATE' + chr(10) + '```' + chr(10) + 'more</parameter></function>'
        for mark in ("```", "~~~"):
            call = '<tool_call>' + body.replace("```", mark) + '</tool_call>'
            with self.subTest(mark=mark):
                text = mark + chr(10) + call + chr(10) + mark + chr(10)
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(text)

    def test_an_offered_call_with_a_fence_line_cannot_hide_a_later_call(self):
        offered = '<tool_call><function=task><parameter=code>' + chr(10) + '```' + chr(10) + '</parameter></function></tool_call>'
        unoffered = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        text = '```' + chr(10) + offered + chr(10) + unoffered + chr(10) + '```' + chr(10)
        with self.assertRaises(self.module.ModelOutputError):
            self.guard().parse(text)

    def test_a_call_after_a_fence_mark_with_a_backtick_is_rejected_too(self):
        unoffered = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        offered = '<tool_call>{"name":"task","arguments":{"v":"`"}}</tool_call>'
        for text in (f"``` {offered}\n{unoffered}\n```\n", f"``` {offered.replace('`', '1')}\n{unoffered}\n```\n"):
            with self.assertRaises(self.module.ModelOutputError):
                self.guard().parse(text)

    def test_stream_holds_every_opener_and_parse_decides(self):
        call = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for text in (f"```\n{call}\n```\n", f'"{call}"', f"`{call}`", f"> {call}", f"    {call}"):
            with self.subTest(text=text):
                guard = self.guard()
                for size in range(1, len(text) + 1):
                    shown = guard.stream(text[:size])
                    self.assertNotIn("PRIVATE", shown)
                    self.assertNotIn("<tool_call>", shown)

    def test_dsml_parameter_literal_tool_call_is_not_a_top_level_call(self):
        from tensorfold.cuda import reply_text
        text = ('<｜DSML｜ calls><｜DSML｜ invoke name="task">'
                '<｜DSML｜ parameter name="note" string="true">'
                '<tool_call>{"name":"todowrite","arguments":{}}</tool_call>'
                '</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>')
        self.assertEqual(self.guard().parse(text), reply_text.parse_tool_calls(text, self.tools))


if __name__ == "__main__":
    unittest.main()
