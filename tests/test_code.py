import json
import threading

from inference_tracker import _code
from inference_tracker._record import Record


def shared_helper():
    return Record.new("openai", "gpt-x", status="ok")


def prompt_suggestions():
    return shared_helper()


def captions():
    return shared_helper()


class Handler:
    def on_request(self):
        return prompt_suggestions()


def test_the_record_names_the_functions_that_made_it_nearest_first():
    r = Handler().on_request()
    assert r.code[:3] == [
        "test_code.shared_helper",
        "test_code.prompt_suggestions",
        "test_code.Handler.on_request",
    ]
    assert r.to_wire()["code.functions"][0] == "test_code.shared_helper"


def test_one_helper_shared_by_two_features_is_told_apart_by_the_next_function():
    assert prompt_suggestions().code[1] != captions().code[1]


def test_library_and_installed_package_frames_are_skipped():
    r = json.loads("{}", object_hook=lambda d: prompt_suggestions())  # json is the standard library
    assert all(not f.startswith(("json.", "inference_tracker.")) for f in r.code)
    assert r.code[0] == "test_code.shared_helper"


def test_a_finish_record_carries_no_chain():
    assert Record.new("fal", "m", phase="finish").code is None


def test_at_most_five_functions_are_kept():
    def deep(n):
        return deep(n - 1) if n else shared_helper()

    assert len(deep(10).code) == _code.DEPTH


def test_a_thread_sees_its_own_side_only():
    got = []
    t = threading.Thread(target=lambda: got.append(prompt_suggestions()))
    t.start()
    t.join()
    assert got[0].code[:2] == ["test_code.shared_helper", "test_code.prompt_suggestions"]
