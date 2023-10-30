import io

import pysnooper as pysnooper

filter_out_lines = ["Starting var:",
                    "exec(",
                    "Source path:",
                    "globals_dict",
                    "tracer.py",
                    "snooping_inner_function()",
                    'File "<string>"',
                    "run_generated_code",
                    "return function(*args, **kwargs)",
                    "source_line = source[line_no - 1]"]


class MockSourceLoader:

    def __init__(self, source):
        self.source = source

    def get_source(self, module_name):
        return self.source


def run_generated_code(check_program):
    tracing = io.StringIO()
    UNIQUE_KEY = '__loader__'
    globals_dict = {UNIQUE_KEY: MockSourceLoader(check_program)}

    @pysnooper.snoop(output=tracing, color=False, depth=2, relative_time=True, normalize=True)
    def snooping_inner_function():
        exec(check_program, globals_dict, {})

    snooping_inner_function()
    trace_output = tracing.getvalue()
    trace_lines = trace_output.split("\n")

    clean_lines = [line for line in trace_lines if not
    any(substring in line for substring in filter_out_lines)]
    clean_output = "\n".join(clean_lines)

    return clean_output
