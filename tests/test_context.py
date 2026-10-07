import asyncio

import inferledger
from inferledger import CARRY_FIELD, carry, context, current, restore


def test_empty_by_default():
    assert current() == inferledger.Context()
    assert carry() == {}


def test_nested_context_keeps_outer_values_and_resets():
    with context(user_id="u1", task_id="t1"):
        with context(parent_id="req-9"):
            assert current() == inferledger.Context("u1", "t1", "req-9")
        with context(task_id="t2"):
            assert current().task_id == "t2"
            assert current().user_id == "u1"
        assert current() == inferledger.Context("u1", "t1", None)
    assert current() == inferledger.Context()


def test_values_become_short_strings():
    with context(user_id=12345, task_id="  "):
        assert current().user_id == "12345"
        assert current().task_id is None
    with context(user_id="x" * 1000):
        assert current().user_id == "x" * 256


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


def test_carry_and_restore_across_a_request():
    with context(user_id="u1", task_id="t1"):
        request_input = {"prompt_id": 7, CARRY_FIELD: carry()}
    assert current().user_id is None

    # The service on the other side, given the whole input
    with restore(request_input):
        assert current() == inferledger.Context("u1", "t1", None)
    # or just the carried part
    with restore({"user_id": "u1"}):
        assert current().user_id == "u1"


def test_restore_ignores_bad_input():
    for bad in (None, "text", 42, [], {CARRY_FIELD: "oops"}, {"user_id": {"nested": 1}}):
        with restore(bad):
            assert current() == inferledger.Context()
