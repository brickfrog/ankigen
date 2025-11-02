import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin, urlparse
import re
import ipaddress
import socket
from typing import List, Set, Optional, Callable, Tuple
import xml.etree.ElementTree as ET  # Added for Sitemap parsing

from ankigen_core.models import CrawledPage
from ankigen_core.utils import RateLimiter, get_logger
from ankigen_core.logging import logger  # Added
from ankigen_core.exceptions import (
    SecurityError,
)

# Security: Maximum URL length to prevent abuse
MAX_URL_LENGTH = 2048


class SSRFProtectionAdapter(HTTPAdapter):
    """
    Custom HTTP adapter that prevents SSRF attacks by validating
    IP addresses at connection time (prevents DNS rebinding attacks).
    """

    def send(self, request, **kwargs) -> requests.Response:
        """Override send to validate IP before making request."""
        # Parse the URL to get hostname
        parsed = urlparse(request.url)
        hostname = parsed.hostname

        if hostname:
            try:
                # Resolve hostname to IP at request time (prevents DNS rebinding)
                ip_str = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_str)

                # Block private, loopback, link-local, and reserved addresses
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                ):
                    msg = f"SSRF protection: Blocked request to private IP {ip_str} for hostname {hostname}"
                    logger.error(msg)
                    raise SecurityError(msg)
            except (socket.gaierror, ValueError) as e:
                logger.error(
                    f"SSRF protection: DNS resolution failed for {hostname}: {e}"
                )
                raise requests.exceptions.ConnectionError(
                    f"DNS resolution failed for {hostname}"
                )

        return super().send(request, **kwargs)


class WebCrawler:
    def __init__(
        self,
        start_url: str,
        max_depth: int = 2,
        requests_per_second: float = 1.0,
        user_agent: str = "AnkiGenBot/1.0",
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        sitemap_url: Optional[str] = None,  # Added for Sitemap (Task 14.1)
        use_sitemap: bool = False,  # Added for Sitemap (Task 14.1)
    ):
        self.start_url = start_url
        self.parsed_start_url = urlparse(start_url)
        self.base_domain = self.parsed_start_url.netloc
        self.max_depth = max_depth
        self.requests_per_second = requests_per_second
        self.delay = 1.0 / requests_per_second if requests_per_second > 0 else 0
        self.user_agent = user_agent
        self.visited_urls: Set[str] = set()
        self.include_patterns = (
            [re.compile(p) for p in include_patterns] if include_patterns else []
        )
        self.exclude_patterns = (
            [re.compile(p) for p in exclude_patterns] if exclude_patterns else []
        )
        self.sitemap_url = sitemap_url  # Added for Sitemap (Task 14.1)
        self.use_sitemap = use_sitemap  # Added for Sitemap (Task 14.1)
        self.logger = get_logger()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

        # Security: Add SSRF protection adapter to prevent DNS rebinding attacks
        # Performance: Configure connection pooling (10 connections per host, 20 total)
        ssrf_adapter = SSRFProtectionAdapter(pool_connections=10, pool_maxsize=20)
        self.session.mount("http://", ssrf_adapter)
        self.session.mount("https://", ssrf_adapter)

        self.rate_limiter = RateLimiter(self.requests_per_second)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False

    def close(self) -> None:
        """Close the requests session and cleanup resources."""
        if hasattr(self, "session") and self.session:
            self.session.close()
            self.logger.debug("WebCrawler session closed")

    def _is_valid_url(self, url: str) -> bool:
        """
        Checks if the URL is valid for crawling with SSRF protection.
        Validates scheme, domain, patterns, and blocks private IP ranges.
        """
        try:
            # Security: URL length check
            if len(url) > MAX_URL_LENGTH:
                logger.warning(
                    f"URL exceeds maximum length ({MAX_URL_LENGTH}): {url[:100]}..."
                )
                return False

            parsed_url = urlparse(url)

            # Security: Protocol whitelist (http/https only)
            if not parsed_url.scheme or parsed_url.scheme.lower() not in [
                "http",
                "https",
            ]:
                logger.debug(f"Invalid scheme for URL: {url}")
                return False

            # Security: SSRF protection - block private IP ranges
            hostname = parsed_url.hostname
            if not hostname:
                logger.warning(f"URL missing hostname: {url}")
                return False

            # Resolve hostname to IP and check if it's private
            try:
                # Get IP address for hostname
                ip_str = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(ip_str)

                # Block private, loopback, link-local, and reserved addresses
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                ):
                    logger.error(
                        f"SSRF protection: Blocked private/internal IP {ip_str} for hostname {hostname}"
                    )
                    return False

            except (socket.gaierror, ValueError, OSError) as e:
                # DNS resolution failed or invalid IP
                logger.warning(f"Could not resolve hostname {hostname}: {e}")
                return False

            # Domain check
            if parsed_url.netloc != self.base_domain:
                logger.debug(f"URL {url} not in base domain {self.base_domain}")
                return False

            # Check include patterns
            if self.include_patterns and not any(
                p.search(url) for p in self.include_patterns
            ):
                logger.debug(f"URL {url} did not match any include patterns.")
                return False

            # Check exclude patterns
            if self.exclude_patterns and any(
                p.search(url) for p in self.exclude_patterns
            ):
                logger.debug(f"URL {url} matched an exclude pattern.")
                return False

        except ValueError:  # Handle potential errors from urlparse on malformed URLs
            logger.warning(f"ValueError when parsing URL: {url}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating URL {url}: {e}", exc_info=True)
            return False

        return True

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extracts, normalizes, and validates links from a BeautifulSoup object.
        """
        found_links: Set[str] = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if not href:  # Skip if href is empty
                continue

            href = href.strip()
            if (
                not href
                or href.startswith("#")
                or href.lower().startswith(("javascript:", "mailto:", "tel:"))
            ):
                continue

            try:
                # Construct absolute URL
                absolute_url = urljoin(base_url, href)

                # Normalize: remove fragment and ensure scheme
                parsed_absolute_url = urlparse(absolute_url)
                normalized_url = parsed_absolute_url._replace(fragment="").geturl()

                # Re-parse to check scheme after normalization, urljoin might produce schemeless if base had none and href was absolute-path-relative
                final_parsed_url = urlparse(normalized_url)
                if not final_parsed_url.scheme:
                    base_parsed_url = urlparse(self.start_url)
                    normalized_url = final_parsed_url._replace(
                        scheme=base_parsed_url.scheme
                    ).geturl()

                if self._is_valid_url(normalized_url):
                    found_links.add(normalized_url)
            except ValueError as e:
                logger.warning(
                    f"Skipping malformed link {href} from base {base_url}: {e}",
                    exc_info=False,
                )
                continue

        return list(found_links)

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extracts and cleans text content from a BeautifulSoup object.
        """
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text

    # --- Sitemap Processing Methods (Task 14.1) ---
    def _fetch_sitemap_content(self, sitemap_url: str) -> Optional[str]:
        """Fetches the content of a given sitemap URL."""
        self.logger.info(f"Fetching sitemap content from: {sitemap_url}")
        try:
            response = self.session.get(sitemap_url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
            return None

    def _parse_sitemap(self, sitemap_content: str) -> List[str]:
        """Parses XML sitemap content and extracts URLs. Handles sitemap indexes."""
        urls: List[str] = []
        try:
            root = ET.fromstring(sitemap_content)

            # Check for sitemap index
            if root.tag.endswith("sitemapindex"):
                self.logger.info("Sitemap index detected. Processing sub-sitemaps.")
                for sitemap_element in root.findall(".//{*}sitemap"):
                    loc_element = sitemap_element.find("{*}loc")
                    if loc_element is not None and loc_element.text:
                        sub_sitemap_url = loc_element.text.strip()
                        self.logger.info(f"Found sub-sitemap: {sub_sitemap_url}")
                        sub_sitemap_content = self._fetch_sitemap_content(
                            sub_sitemap_url
                        )
                        if sub_sitemap_content:
                            urls.extend(self._parse_sitemap(sub_sitemap_content))
            # Process regular sitemap
            elif root.tag.endswith("urlset"):
                for url_element in root.findall(".//{*}url"):
                    loc_element = url_element.find("{*}loc")
                    if loc_element is not None and loc_element.text:
                        urls.append(loc_element.text.strip())
            else:
                self.logger.warning(f"Unknown root tag in sitemap: {root.tag}")

        except ET.ParseError as e:
            self.logger.error(f"Error parsing sitemap XML: {e}")
        return list(set(urls))  # Return unique URLs

    def _get_urls_from_sitemap(self) -> List[str]:
        """Fetches and parses the sitemap to get a list of URLs."""
        if not self.sitemap_url:
            self.logger.warning(
                "Sitemap URL is not provided. Cannot fetch URLs from sitemap."
            )
            return []

        sitemap_content = self._fetch_sitemap_content(self.sitemap_url)
        if not sitemap_content:
            return []

        sitemap_urls = self._parse_sitemap(sitemap_content)
        self.logger.info(f"Extracted {len(sitemap_urls)} unique URLs from sitemap(s).")
        return sitemap_urls

    # --- End Sitemap Processing Methods ---

    def _initialize_crawl_queue(self) -> List[Tuple[str, int, Optional[str]]]:
        """Initialize the crawl queue from sitemap or start URL.

        Returns:
            List of tuples (url, depth, parent_url) to visit
        """
        urls_to_visit: List[Tuple[str, int, Optional[str]]] = []

        if self.use_sitemap and self.sitemap_url:
            self.logger.info(f"Attempting to use sitemap: {self.sitemap_url}")
            sitemap_extracted_urls = self._get_urls_from_sitemap()
            if sitemap_extracted_urls:
                for url in sitemap_extracted_urls:
                    if self._is_valid_url(url):
                        urls_to_visit.append((url, 0, None))
                self.logger.info(
                    f"Initialized {len(urls_to_visit)} URLs to visit from sitemap after validation."
                )
            else:
                self.logger.warning(
                    "Sitemap processing yielded no URLs. Falling back to start_url."
                )
                if self._is_valid_url(self.start_url):
                    urls_to_visit.append((self.start_url, 0, None))
        else:
            if self._is_valid_url(self.start_url):
                urls_to_visit.append((self.start_url, 0, None))

        return urls_to_visit

    def _extract_page_metadata(
        self, soup: BeautifulSoup, url: str
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """Extract title, meta description, and meta keywords from page.

        Args:
            soup: BeautifulSoup object of the page
            url: URL being processed (for logging)

        Returns:
            Tuple of (title, meta_description, meta_keywords_list)
        """
        # Extract title
        page_title_tag = soup.find("title")
        page_title: Optional[str] = None
        if isinstance(page_title_tag, Tag) and page_title_tag.string:
            page_title = page_title_tag.string.strip()
        else:
            self.logger.debug(f"No title tag found for {url}")

        # Extract meta description
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_description: Optional[str] = None
        if isinstance(meta_desc_tag, Tag):
            content = meta_desc_tag.get("content")
            if isinstance(content, str):
                meta_description = content.strip()
            elif isinstance(content, list):
                meta_description = " ".join(str(item) for item in content).strip()
                self.logger.debug(
                    f"Meta description for {url} was a list, joined: {meta_description}"
                )
        else:
            self.logger.debug(f"No meta description found for {url}")

        # Extract meta keywords
        meta_keywords_tag = soup.find("meta", attrs={"name": "keywords"})
        meta_keywords: List[str] = []
        if isinstance(meta_keywords_tag, Tag):
            content_kw = meta_keywords_tag.get("content")
            raw_keywords_content: str = ""
            if isinstance(content_kw, str):
                raw_keywords_content = content_kw
            elif isinstance(content_kw, list):
                raw_keywords_content = " ".join(str(item) for item in content_kw)
                self.logger.debug(
                    f"Meta keywords for {url} was a list, joined: {raw_keywords_content}"
                )

            if raw_keywords_content:
                meta_keywords = [
                    k.strip() for k in raw_keywords_content.split(",") if k.strip()
                ]
        else:
            self.logger.debug(f"No meta keywords found for {url}")

        return page_title, meta_description, meta_keywords

    def _should_skip_url(self, url: str, depth: int) -> Tuple[bool, Optional[str]]:
        """Check if URL should be skipped.

        Args:
            url: URL to check
            depth: Current depth of URL

        Returns:
            Tuple of (should_skip, skip_reason)
        """
        if url in self.visited_urls:
            return True, f"Skipped (visited): {url}"

        if depth > self.max_depth:
            logger.debug(
                f"Skipping URL {url} due to depth {depth} > max_depth {self.max_depth}"
            )
            return True, f"Skipped (max depth): {url}"

        return False, None

    def crawl(
        self, progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[CrawledPage]:
        # Initialize URLs using helper method
        urls_to_visit = self._initialize_crawl_queue()
        crawled_pages: List[CrawledPage] = []
        initial_total_for_progress = len(urls_to_visit)

        processed_count = 0
        while urls_to_visit:
            current_url, current_depth, current_parent_url = urls_to_visit.pop(0)

            current_total_for_progress = (
                initial_total_for_progress
                if self.use_sitemap
                else processed_count + len(urls_to_visit) + 1
            )

            if progress_callback:
                progress_callback(
                    processed_count,
                    current_total_for_progress,
                    current_url,
                )

            # Check if URL should be skipped using helper method
            should_skip, skip_reason = self._should_skip_url(current_url, current_depth)
            if should_skip:
                if progress_callback and skip_reason:
                    dynamic_total = (
                        initial_total_for_progress
                        if self.use_sitemap
                        else processed_count + len(urls_to_visit) + 1
                    )
                    progress_callback(processed_count, dynamic_total, skip_reason)
                continue

            self.logger.info(
                f"Crawling (Depth {current_depth}): {current_url} ({processed_count + 1}/{current_total_for_progress})"
            )

            if progress_callback:
                progress_callback(
                    processed_count, current_total_for_progress, current_url
                )

            self.visited_urls.add(current_url)

            self.rate_limiter.wait()

            try:
                response = self.session.get(current_url, timeout=10)
                response.raise_for_status()
                html_content = response.text
                soup = BeautifulSoup(html_content, "html.parser")

                # Extract metadata using helper method
                page_title, meta_description, meta_keywords = (
                    self._extract_page_metadata(soup, current_url)
                )

                text_content = self._extract_text(soup)

                page_data = CrawledPage(
                    url=current_url,
                    html_content=html_content,
                    text_content=text_content,
                    title=page_title,
                    meta_description=meta_description,
                    meta_keywords=meta_keywords,
                    crawl_depth=current_depth,
                    parent_url=current_parent_url,
                )
                crawled_pages.append(page_data)
                self.logger.info(f"Successfully processed and stored: {current_url}")

                if current_depth < self.max_depth:
                    found_links = self._extract_links(soup, current_url)
                    self.logger.debug(
                        f"Found {len(found_links)} links on {current_url}"
                    )
                    for link in found_links:
                        if link not in self.visited_urls:
                            urls_to_visit.append((link, current_depth + 1, current_url))

            except requests.exceptions.HTTPError as e:
                self.logger.error(
                    f"HTTPError for {current_url}: {e.response.status_code} - {e.response.reason}. Response: {e.response.text[:200]}...",
                    exc_info=False,
                )
                processed_count += 1
            except requests.exceptions.ConnectionError as e:
                self.logger.error(
                    f"ConnectionError for {current_url}: {e}", exc_info=False
                )
                processed_count += 1
            except requests.exceptions.Timeout as e:
                self.logger.error(f"Timeout for {current_url}: {e}", exc_info=False)
                processed_count += 1
            except requests.exceptions.RequestException as e:
                self.logger.error(
                    f"RequestException for {current_url}: {e}", exc_info=True
                )
                processed_count += 1
            except Exception as e:
                self.logger.error(
                    f"An unexpected error occurred while processing {current_url}: {e}",
                    exc_info=True,
                )
                processed_count += 1

        self.logger.info(
            f"Crawl completed. Total pages processed/attempted: {processed_count}. Successfully crawled pages: {len(crawled_pages)}"
        )
        if progress_callback:
            progress_callback(processed_count, processed_count, "Crawling complete.")

        return crawled_pages
