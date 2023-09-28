import argparse
import asyncio
import logging
import os

from algo.ai_handler import AiHandler
from algo.ai_invoker import retry_with_fallback_models

import asyncio


class SimplePrompt:
    def __init__(self):
        self.ai_handler = AiHandler()

    async def run(self, model):
        response, finish_reason = await self.ai_handler.chat_completion(
            model=model,
            temperature=0.2,
            system="",
            user="what is the capital of the united states"
        )

        return response


def main():
    loop = asyncio.get_event_loop()
    p = SimplePrompt()
    loop.run_until_complete(retry_with_fallback_models(p.run))
    loop.close()


if __name__ == '__main__':
    main()
