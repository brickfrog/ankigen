"""Context7 integration for library documentation"""

import asyncio
import subprocess
import json
import re
from typing import Optional, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from ankigen_core.logging import logger
from ankigen_core.exceptions import (
    ValidationError,
)

# Security: Whitelist pattern for library names and topics
# Allows: letters, numbers, hyphens, underscores, dots, forward slashes, @scopes
SAFE_LIBRARY_PATTERN = re.compile(r"^[@a-zA-Z0-9._/-]+$")
SAFE_TOPIC_PATTERN = re.compile(r"^[a-zA-Z0-9\s.,_-]+$")
MAX_STRING_LENGTH = 200  # Prevent excessively long inputs
SUBPROCESS_TIMEOUT = 60.0  # 60 second timeout for Context7 calls


class Context7Client:
    """Context7 MCP client for fetching library documentation"""

    def __init__(self):
        pass  # No state needed - each call creates fresh subprocess

    @staticmethod
    def _validate_library_name(library_name: str) -> bool:
        """Validate library name to prevent injection attacks"""
        if not library_name or len(library_name) > MAX_STRING_LENGTH:
            return False
        return SAFE_LIBRARY_PATTERN.match(library_name) is not None

    @staticmethod
    def _validate_topic(topic: str) -> bool:
        """Validate topic string to prevent injection attacks"""
        if not topic or len(topic) > MAX_STRING_LENGTH:
            return False
        return SAFE_TOPIC_PATTERN.match(topic) is not None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def call_context7_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a Context7 tool via direct JSONRPC with retry logic"""
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

            # Send both requests with timeout protection
            # Optimize: Use list join for string concatenation
            input_data = "\n".join([json.dumps(init_request), json.dumps(request), ""])
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode()),
                    timeout=SUBPROCESS_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # Proper process cleanup on timeout
                try:
                    if process.returncode is None:  # Process still running
                        process.kill()
                        # Wait for process to actually terminate
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                except Exception as cleanup_error:
                    logger.error(f"Error during process cleanup: {cleanup_error}")
                raise TimeoutError(
                    f"Context7 subprocess timed out after {SUBPROCESS_TIMEOUT}s"
                )
            except Exception:
                # Clean up process on any other error
                try:
                    if process.returncode is None:
                        process.kill()
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                except Exception:
                    pass  # Best effort cleanup
                raise

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
        # Security: Validate library name to prevent injection
        if not self._validate_library_name(library_name):
            logger.error(f"Invalid library name (security): '{library_name}'")
            raise ValidationError(
                f"Invalid library name: must match pattern {SAFE_LIBRARY_PATTERN.pattern}"
            )

        logger.info(f"Resolving library ID for: {library_name}")

        result = await self.call_context7_tool(
            "resolve-library-id", {"libraryName": library_name}
        )

        if result and result.get("success") and result.get("text"):
            text = result["text"]

            # Parse the structured response format
            libraries = []
            lines = text.split("\n")

            current_lib = {}
            for line in lines:
                line = line.strip()

                # Parse title
                if line.startswith("- Title:"):
                    if current_lib and current_lib.get("id"):
                        libraries.append(current_lib)
                    current_lib = {
                        "title": line.replace("- Title:", "").strip().lower()
                    }

                # Parse library ID
                elif line.startswith("- Context7-compatible library ID:"):
                    lib_id = line.replace(
                        "- Context7-compatible library ID:", ""
                    ).strip()
                    if current_lib is not None:
                        current_lib["id"] = lib_id

                # Parse code snippets count
                elif line.startswith("- Code Snippets:"):
                    snippets_str = line.replace("- Code Snippets:", "").strip()
                    try:
                        snippets = int(snippets_str)
                        if current_lib is not None:
                            current_lib["snippets"] = snippets
                    except ValueError:
                        pass

                # Parse trust score
                elif line.startswith("- Trust Score:"):
                    score_str = line.replace("- Trust Score:", "").strip()
                    try:
                        trust = float(score_str)
                        if current_lib is not None:
                            current_lib["trust"] = trust
                    except ValueError:
                        pass

            # Add the last library if exists
            if current_lib and current_lib.get("id"):
                libraries.append(current_lib)

            # If we found libraries, pick the best match
            if libraries:
                search_term = library_name.lower()

                # Score each library
                best_lib = None
                best_score = -1

                for lib in libraries:
                    score = 0
                    lib_title = lib.get("title", "")
                    lib_id = lib["id"].lower()

                    # Exact title match gets highest priority
                    if lib_title == search_term:
                        score += 10000
                    # Check if it's exactly "pandas" in the path (not geopandas, etc)
                    elif lib_id == f"/{search_term}-dev/{search_term}":
                        score += 5000
                    elif f"/{search_term}/" in lib_id or lib_id.endswith(
                        f"/{search_term}"
                    ):
                        score += 2000
                    # Partial title match (but penalize if it's a compound like "geopandas")
                    elif search_term in lib_title:
                        if lib_title == search_term:
                            score += 1000
                        elif lib_title.startswith(search_term):
                            score += 200
                        else:
                            score += 50

                    # Strong bonus for code snippets (indicates main library)
                    snippets = lib.get("snippets", 0)
                    score += snippets / 10  # Pandas has 7386 snippets

                    # Significant bonus for trust score (high trust = official/authoritative)
                    trust = lib.get("trust", 0)
                    score += trust * 100  # Trust 9.2 = 920 points, Trust 7 = 700 points

                    # Debug logging
                    if search_term in lib_title or search_term in lib_id:
                        logger.debug(
                            f"Scoring {lib['id']}: title='{lib_title}', snippets={snippets}, "
                            f"trust={trust}, score={score:.2f}"
                        )

                    if score > best_score:
                        best_score = score
                        best_lib = lib

                if best_lib:
                    logger.info(
                        f"Resolved '{library_name}' to ID: {best_lib['id']} "
                        f"(title: {best_lib.get('title', 'unknown')}, snippets: {best_lib.get('snippets', 0)}, "
                        f"trust: {best_lib.get('trust', 0)}, score: {best_score:.2f})"
                    )
                    return best_lib["id"]

        logger.warning(f"Could not resolve library ID for '{library_name}'")
        return None

    async def get_library_docs(
        self, library_id: str, topic: Optional[str] = None, tokens: int = 5000
    ) -> Optional[str]:
        """Get documentation for a library"""
        # Security: Validate library_id (should start with /)
        if (
            not library_id
            or not library_id.startswith("/")
            or len(library_id) > MAX_STRING_LENGTH
        ):
            logger.error(f"Invalid library ID format (security): '{library_id}'")
            raise ValidationError("Invalid library ID format")

        # Security: Validate topic if provided
        if topic and not self._validate_topic(topic):
            logger.error(f"Invalid topic (security): '{topic}'")
            raise ValidationError(
                f"Invalid topic: must match pattern {SAFE_TOPIC_PATTERN.pattern}"
            )

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


async def test_context7() -> None:
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
