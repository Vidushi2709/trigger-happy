"""Run: python src/test_evaluate.py"""
from evaluate import correct, parse_calls, schema

call = lambda name, args: f'<tool_call>\n{{"name": "{name}", "arguments": {args}}}\n</tool_call>'
tri = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]

assert correct(parse_calls(call("calculate_triangle_area", '{"base": 10, "height": 5}')), tri)  # optional omitted
assert correct(parse_calls(call("calculate_triangle_area", '{"base": 10.0, "height": 5, "unit": "Units"}')), tri)  # int=float, case
assert not correct(parse_calls(call("calculate_triangle_area", '{"base": 10}')), tri)  # required missing
assert not correct(parse_calls(call("calculate_triangle_area", '{"base": 10, "height": 5, "color": "red"}')), tri)  # extra arg
assert not correct(parse_calls(call("other", '{"base": 10, "height": 5}')), tri)  # wrong function
assert not correct(parse_calls("<tool_call>{not json}</tool_call>"), tri)  # unparseable

par = [{"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}}, {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}}]
two = call("spotify.play", '{"artist": "Maroon 5", "duration": 15}') + call("spotify.play", '{"artist": "taylor swift", "duration": 20}')
assert correct(parse_calls(two), par)  # any order
assert not correct(parse_calls(two.split("</tool_call>")[0] + "</tool_call>"), par)  # one call short

assert correct(parse_calls("I can't help with that using these tools."), [])  # irrelevance: no call = right
assert not correct(parse_calls(call("x", "{}")), [])

assert schema({"type": "dict", "properties": {"a": {"type": "float"}}}) == {"type": "object", "properties": {"a": {"type": "number"}}}
print("ok")
