import asyncio
import functools
import logging
import os
import re

import numpy as np
import yaml
from jinja2 import Environment, StrictUndefined

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.code_contests.eval.code_test_runners import eval_solution
from alpha_codium.config_loader import get_settings
from alpha_codium.llm.ai_handler import AiHandler
from alpha_codium.llm.ai_invoker import retry_with_fallback_models
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_baseline(self, problem):
    try:
        logging.info("Using baseline prompt")
        f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_baseline")
        response_baseline, _ = await retry_with_fallback_models(f)
        recent_solution = self.postprocess_response(response_baseline)
        return recent_solution
    except Exception as e:
        logging.error(f"Error: {e}")
        exit(-1)
