import json

from inferledger import Record, context, error_name


def test_new_record_takes_the_current_context():
    with context(user_id="u1", task_id="t1", parent_id="req-parent"):
        r = Record.new("fal", "fal-ai/flux/dev", request_id="req-1", status="ok")
    assert (r.user_id, r.task_id, r.parent_id) == ("u1", "t1", "req-parent")
    assert len(r.id) == 32


def test_wire_format_uses_otel_names_and_drops_empty_fields():
    r = Record.new(
        "gcp.vertex_ai",
        "gemini-3.1-flash-image",
        request_id="resp-1",
        status="ok",
        duration_s=4.2,
        units={"input_tokens": 120, "output_tokens": 1120},
    )
    wire = r.to_wire()
    assert wire["gen_ai.provider.name"] == "gcp.vertex_ai"
    assert wire["gen_ai.request.model"] == "gemini-3.1-flash-image"
    assert wire["gen_ai.response.id"] == "resp-1"
    assert wire["units"] == {"input_tokens": 120, "output_tokens": 1120}
    assert "user_id" not in wire and "settings" not in wire and "error.type" not in wire
    json.dumps(wire)  # always valid JSON


def test_start_and_finish_records_for_a_queued_call():
    start = Record.new(
        "fal",
        "prod-myapp-wan2-2-img2video",
        phase="start",
        request_id="req-1",
        settings={"resolution": "720p", "duration": 5},
    )
    finish = Record.new(
        "fal", "prod-myapp-wan2-2-img2video", phase="finish", request_id="req-1", status="ok", duration_s=104.8
    )
    assert start.to_wire()["phase"] == "start" and "status" not in start.to_wire()
    assert finish.to_wire()["gen_ai.response.id"] == start.to_wire()["gen_ai.response.id"]


def test_settings_are_kept_as_given_unless_they_cannot_be_sent():
    r = Record.new(
        "gcp.vertex_ai",
        "veo-3.1-fast",
        settings={"resolution": "1080P ", "durationSeconds": 8, "audio": {"on": True}, "client": object()},
        units={"video_seconds": 8, "note": "text", "flag": True, "bad": float("nan")},
    )
    wire = r.to_wire()
    # Any format is kept; only what JSON can't carry is left out.
    assert wire["settings"] == {"resolution": "1080P ", "durationSeconds": 8, "audio": {"on": True}}
    assert wire["units"] == {"video_seconds": 8}
    json.dumps(wire, allow_nan=False)


def test_error_type_is_the_class_name_not_the_message():
    try:
        raise TimeoutError("timed out for prompt: a photo of my kid")
    except TimeoutError as e:
        assert error_name(e) == "TimeoutError"
