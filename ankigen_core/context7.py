"""Context7 integration for library documentation"""

import asyncio
import subprocess
import json
from typing import Optional, Dict, Any
from ankigen_core.logging import logger


class Context7Client:
    """Context7 MCP client for fetching library documentation"""

    def __init__(self):
        self.server_process = None

    async def call_context7_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a Context7 tool via direct JSONRPC"""
        try:
            # Build the JSONRPC request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            }

            # Call the Context7 server
            process = await asyncio.create_subprocess_exec(
                "npx",
                "@upstash/context7-mcp",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Send initialization first
            init_request = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "ankigen", "version": "1.0.0"},
                },
            }

            # Send both requests
            input_data = json.dumps(init_request) + "\n" + json.dumps(request) + "\n"
            stdout, stderr = await process.communicate(input=input_data.encode())

            # Parse responses
            responses = stdout.decode().strip().split("\n")
            if len(responses) >= 2:
                # Skip init response, get tool response
                tool_response = json.loads(responses[1])

                if "result" in tool_response:
                    result = tool_response["result"]
                    # Extract content from the result
                    if "content" in result and result["content"]:
                        content_item = result["content"][0]
                        if "text" in content_item:
                            return {"text": content_item["text"], "success": True}
                        elif "type" in content_item and content_item["type"] == "text":
                            return {
                                "text": content_item.get("text", ""),
                                "success": True,
                            }
                    return {"error": "No content in response", "success": False}
                elif "error" in tool_response:
                    return {"error": tool_response["error"], "success": False}

            return {"error": "Invalid response format", "success": False}

        except Exception as e:
            logger.error(f"Error calling Context7 tool {tool_name}: {e}")
            return {"error": str(e), "success": False}

    async def resolve_library_id(self, library_name: str) -> Optional[str]:
        """Resolve a library name to a Context7-compatible ID"""
        logger.info(f"Resolving library ID for: {library_name}")

        result = await self.call_context7_tool(
            "resolve-library-id", {"libraryName": library_name}
        )

        if result and result.get("success") and result.get("text"):
            # Parse the text to extract library ID
            text = result["text"]
            import re

            # First, look for specific Context7-compatible library ID mentions
            lines = text.split("\n")
            for line in lines:
                if "Context7-compatible library ID:" in line:
                    # Extract the ID after the colon
                    parts = line.split("Context7-compatible library ID:")
                    if len(parts) > 1:
                        library_id = parts[1].strip()
                        if library_id.startswith("/"):
                            logger.info(
                                f"Resolved '{library_name}' to ID: {library_id}"
                            )
                            return library_id

            # Fallback: Look for library ID pattern but be more specific
            # Must have actual library names, not generic /org/project
            matches = re.findall(r"/[\w-]+/[\w.-]+(?:/[\w.-]+)?", text)
            for match in matches:
                # Filter out generic placeholders
                if match != "/org/project" and "example" not in match.lower():
                    logger.info(f"Resolved '{library_name}' to ID: {match}")
                    return match

        logger.warning(f"Could not resolve library ID for '{library_name}'")
        return None

    async def get_library_docs(
        self, library_id: str, topic: Optional[str] = None, tokens: int = 5000
    ) -> Optional[str]:
        """Get documentation for a library"""
        logger.info(
            f"Fetching docs for: {library_id}" + (f" (topic: {topic})" if topic else "")
        )

        args = {"context7CompatibleLibraryID": library_id, "tokens": tokens}
        if topic:
            args["topic"] = topic

        result = await self.call_context7_tool("get-library-docs", args)

        if result and result.get("success") and result.get("text"):
            docs = result["text"]
            logger.info(f"Retrieved {len(docs)} characters of documentation")
            return docs

        logger.warning(f"Could not fetch docs for '{library_id}'")
        return None

    async def fetch_library_documentation(
        self, library_name: str, topic: Optional[str] = None, tokens: int = 5000
    ) -> Optional[str]:
        """Convenience method to resolve and fetch docs in one call"""
        library_id = await self.resolve_library_id(library_name)
        if not library_id:
            return None

        return await self.get_library_docs(library_id, topic, tokens)


async def test_context7():
    """Test the Context7 integration"""
    client = Context7Client()

    print("Testing Context7 integration...")

    # Test resolving a library
    library_id = await client.resolve_library_id("react")
    if library_id:
        print(f"✓ Resolved 'react' to ID: {library_id}")

        # Test fetching docs
        docs = await client.get_library_docs(library_id, topic="hooks", tokens=2000)
        if docs:
            print(f"✓ Fetched {len(docs)} characters of documentation")
            print(f"Preview: {docs[:300]}...")
        else:
            print("✗ Failed to fetch documentation")
    else:
        print("✗ Failed to resolve library ID")


if __name__ == "__main__":
    asyncio.run(test_context7())
