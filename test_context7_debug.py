"""Debug Context7 response to understand format"""

import asyncio
from ankigen_core.context7 import Context7Client


async def test_debug():
    client = Context7Client()

    # Get raw response
    result = await client.call_context7_tool(
        "resolve-library-id", {"libraryName": "pandas"}
    )

    if result and result.get("success") and result.get("text"):
        print("=== RAW RESPONSE ===")
        print(result["text"])
        print("=== END RESPONSE ===")

        # Also show line by line with indices
        lines = result["text"].split("\n")
        print("\n=== LINES WITH INDICES ===")
        for i, line in enumerate(lines):
            print(f"{i:3}: '{line}'")
    else:
        print("Failed to get response:", result)


if __name__ == "__main__":
    asyncio.run(test_debug())
