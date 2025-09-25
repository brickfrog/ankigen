"""Test pandas library resolution specifically"""

import asyncio
from ankigen_core.context7 import Context7Client


async def test_pandas():
    client = Context7Client()
    library_id = await client.resolve_library_id("pandas")
    print(f"Result: {library_id}")


if __name__ == "__main__":
    asyncio.run(test_pandas())
