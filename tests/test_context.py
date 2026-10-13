import asyncio

import inference_tracker
from inference_tracker import Record, context, current


def test_empty_by_default():
    assert current() == inference_tracker.Context()


def test_nested_context_keeps_outer_values_and_resets():
    with context(user_id="u1", task_id="t1"):
        with context(parent_provider="fal", parent_request_id="req-9"):
            assert current() == inference_tracker.Context("u1", "t1", "fal", "req-9")
            with context(parent_request_id="req-10"):
                assert current() == inference_tracker.Context("u1", "t1", "fal", "req-10")
        with context(task_id="t2"):
            assert current().task_id == "t2"
            assert current().user_id == "u1"
        assert current() == inference_tracker.Context("u1", "t1")
    assert current() == inference_tracker.Context()


def test_values_become_short_strings():
    with context(user_id=12345, task_id="  "):
        assert current().user_id == "12345"
        assert current().task_id is None
    with context(user_id="x" * 1000, parent_request_id=" req-1 "):
        assert current().user_id == "x" * 256
        assert current().parent_request_id == "req-1"


def test_concurrent_requests_keep_their_own_context():
    # Like a fal app serving 3 requests at once on one machine.
    async def request(uid):
        with context(user_id=uid):
            await asyncio.sleep(0.01)
            first = current().user_id
            await asyncio.sleep(0.01)
            # asyncio.to_thread copies the context into the thread
            in_thread = await asyncio.to_thread(lambda: current().user_id)
            return first, in_thread

    async def main():
        return await asyncio.gather(*(request(f"u{i}") for i in range(3)))

    assert asyncio.run(main()) == [("u0", "u0"), ("u1", "u1"), ("u2", "u2")]


def test_a_call_made_inside_another_names_it_as_parent():
    # One of our fal apps, serving fal's request req-9, calls a model of its own.
    with context(parent_provider="fal", parent_request_id="req-9"):
        inner = Record.new("gcp.vertex_ai", "veo-3.1-fast", request_id="op-1")
    assert (inner.parent_provider, inner.parent_request_id) == ("fal", "req-9")
    assert inner.user_id is None  # the server finds the user by walking up to the parent
