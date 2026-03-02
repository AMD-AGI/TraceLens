#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
SharePoint Trace Loader Module
Handles downloading trace files from SharePoint URLs with authentication
"""

import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

logger = logging.getLogger(__name__)

try:
    import msal
    from office365.sharepoint.client_context import ClientContext
    from office365.sharepoint.files.file import File

    _HAS_SHAREPOINT_DEPS = True
except ImportError:
    msal = None
    ClientContext = None
    File = None
    _HAS_SHAREPOINT_DEPS = False


class TokenWrapper:
    """Wrapper for MSAL token dictionary to match office365 expectations."""

    def __init__(self, token_dict):
        self.accessToken = token_dict["access_token"]
        self.tokenType = token_dict.get("token_type", "Bearer")


def _acquire_sharepoint_token():
    """
    Acquire SharePoint access token using device code flow with persistent caching.

    Returns:
        Dict containing access token and metadata

    Raises:
        ImportError: If SharePoint dependencies are not installed
        ValueError: If device flow creation fails
        Exception: If token acquisition fails
    """
    if not _HAS_SHAREPOINT_DEPS:
        raise ImportError(
            "SharePoint dependencies not installed. "
            "Please install with: pip install office365-rest-python-client msal"
        )

    # Microsoft Office Client ID
    client_id = "d3590ed6-52b3-4102-aeff-aad2292ab01c"
    tenant = "amdcloud.onmicrosoft.com"
    authority_url = f"https://login.microsoftonline.com/{tenant}"

    # Scope for SharePoint
    resource_url = "https://amdcloud.sharepoint.com"
    scopes = [f"{resource_url}/.default"]

    # Setup persistent cache
    cache_dir = Path.home() / ".cache" / "tracelens_jarvis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "sharepoint_token_cache.bin"

    cache = msal.SerializableTokenCache()
    if cache_file.exists():
        with open(cache_file, "r") as f:
            cache.deserialize(f.read())

    # Create app with cache
    app = msal.PublicClientApplication(
        client_id, authority=authority_url, token_cache=cache
    )

    # 1. Try to acquire token silently (from cache/refresh token)
    accounts = app.get_accounts()
    result = None
    if accounts:
        logger.info("Attempting to acquire token silently from cache...")
        result = app.acquire_token_silent(scopes, account=accounts[0])

    # 2. If silent fails, fall back to interactive Device Code Flow
    if not result:
        logger.info("No cached token found. Initiating device code flow...")
        # Initiate Device Code Flow
        flow = app.initiate_device_flow(scopes=scopes)
        if "user_code" not in flow:
            raise ValueError(
                f"Failed to create device flow. "
                f"Error: {flow.get('error')}, "
                f"Description: {flow.get('error_description')}"
            )

        print("\n" + "=" * 60)
        print(flow["message"])
        print("=" * 60 + "\n")
        sys.stdout.flush()

        # Block until user logs in
        result = app.acquire_token_by_device_flow(flow)

    if "access_token" in result:
        # Save cache to disk
        if cache.has_state_changed:
            with open(cache_file, "w") as f:
                f.write(cache.serialize())
        logger.info("Successfully acquired SharePoint access token")
        return result
    else:
        raise Exception(f"Could not acquire token: {result.get('error_description')}")


def load_trace_from_sharepoint(url: str, local_path: Optional[Path] = None) -> str:
    """
    Download a trace file from SharePoint URL to local path.

    Args:
        url: SharePoint URL to the trace file (direct link or sharing link)
        local_path: Optional local path to save the file. If None, uses temp file.

    Returns:
        str: Absolute path to the downloaded file

    Raises:
        ImportError: If SharePoint dependencies are not installed
        ValueError: If URL format is invalid or points to a folder
        Exception: If download fails
    """
    if not _HAS_SHAREPOINT_DEPS:
        raise ImportError(
            "SharePoint dependencies not installed. "
            "Please install with: pip install office365-rest-python-client msal"
        )

    # Parse site URL from the full file URL
    parsed = urlparse(url)
    path_str = unquote(parsed.path)

    # Check if this is a download.aspx URL with UniqueId parameter
    if "download.aspx" in url.lower() and "uniqueid=" in url.lower():
        from urllib.parse import parse_qs

        query_params = parse_qs(parsed.query)
        if "UniqueId" in query_params or "uniqueid" in query_params:
            unique_id = query_params.get("UniqueId", query_params.get("uniqueid"))[0]
            logger.info(f"Detected download.aspx URL with UniqueId: {unique_id}")
            # For download.aspx URLs, we need to use the GetFileByUniqueId API
            # Extract site from path (typically /sites/SiteName/)
            path_parts = parsed.path.split("/")
            if len(path_parts) > 2 and path_parts[1] == "sites":
                site_name = path_parts[2]
                site_url = f"{parsed.scheme}://{parsed.netloc}/sites/{site_name}"
            else:
                site_url = f"{parsed.scheme}://{parsed.netloc}"

            logger.info(f"Connecting to SharePoint site: {site_url}")
            token = _acquire_sharepoint_token()
            ctx = ClientContext(site_url).with_access_token(lambda: TokenWrapper(token))
            ctx.authentication_context._token_expires = datetime.now(
                timezone.utc
            ) + timedelta(hours=1)

            # Use GetFileByUniqueId to get the file
            logger.info(f"Fetching file by UniqueId: {unique_id}")
            from office365.sharepoint.files.file import File as SPFile

            file_obj = ctx.web.get_file_by_id(unique_id)
            ctx.load(file_obj)
            ctx.execute_query()

            # Get filename from file object - need to access the Name property after loading
            filename = file_obj.name
            if not filename:
                # Fallback if name is not available
                filename = f"file_{unique_id}.json"
            logger.info(f"Retrieved filename: {filename}")

            # Create local path
            if local_path is None:
                local_path = Path(tempfile.mkdtemp()) / filename
            else:
                local_path = Path(local_path)
                if local_path.is_dir() or (
                    not local_path.exists() and not local_path.suffix
                ):
                    local_path = local_path / filename

            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            logger.info(f"Downloading file to: {local_path}")
            file_content = file_obj.read()
            ctx.execute_query()

            with open(local_path, "wb") as f:
                f.write(file_content)

            logger.info(f"Successfully downloaded to: {local_path}")
            return str(local_path.absolute())

    # Check if this is a web view URL with 'id' parameter containing the actual path
    if "AllItems.aspx" in url and "id=" in url:
        from urllib.parse import parse_qs

        query_params = parse_qs(parsed.query)
        if "id" in query_params:
            # Extract the file/folder path from the 'id' parameter
            path_str = unquote(query_params["id"][0])
            logger.info(f"Extracted path from 'id' parameter: {path_str}")

    # Remove SharePoint sharing prefixes if present
    # e.g. /:u:/r/sites/... -> /sites/...
    # e.g. /:f:/r/sites/... -> /sites/... (folder link)
    for prefix in ["/:u:/r", "/:f:/r", "/:x:/r", "/:v:/r"]:
        if path_str.startswith(prefix):
            path_str = path_str.replace(prefix, "", 1)
            break

    # Check if this is a folder path (no file extension)
    path_obj = Path(path_str)
    if not path_obj.suffix:
        raise ValueError(
            f"URL appears to point to a SharePoint folder, not a file.\n"
            f"Path: {path_str}\n\n"
            f"Please navigate to the specific file you want to download.\n"
            f"The URL should end with a filename (e.g., .json, .json.gz, .tar.gz)"
        )

    path_parts = path_str.split("/")

    # Heuristic to find the site URL.
    # Usually it's https://<tenant>.sharepoint.com/sites/<site_name>
    # path_parts[0] is empty string because path starts with /
    if len(path_parts) > 2 and path_parts[1] == "sites":
        site_url = f"{parsed.scheme}://{parsed.netloc}/{path_parts[1]}/{path_parts[2]}"
    else:
        # Fallback to root site if not in /sites/
        site_url = f"{parsed.scheme}://{parsed.netloc}"

    file_server_relative_url = path_str

    # Create local path if not provided or if it's a directory
    if local_path is None:
        filename = Path(path_str).name
        local_path = Path(tempfile.mkdtemp()) / filename
    else:
        local_path = Path(local_path)
        # If it's a directory (or looks like one), extract filename from URL
        if local_path.is_dir() or (not local_path.exists() and not local_path.suffix):
            filename = Path(path_str).name
            local_path = local_path / filename

    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to SharePoint site: {site_url}")
    token = _acquire_sharepoint_token()

    ctx = ClientContext(site_url).with_access_token(lambda: TokenWrapper(token))
    # Fix for datetime comparison error
    ctx.authentication_context._token_expires = datetime.now(timezone.utc) + timedelta(
        hours=1
    )

    logger.info(f"Downloading file from SharePoint: {file_server_relative_url}")
    response = File.open_binary(ctx, file_server_relative_url)

    with open(local_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Successfully downloaded to: {local_path}")
    return str(local_path.absolute())


def is_sharepoint_url(url: str) -> bool:
    """
    Check if a URL is a SharePoint URL.

    Args:
        url: URL to check

    Returns:
        bool: True if URL is a SharePoint URL
    """
    parsed = urlparse(url)
    return "sharepoint.com" in parsed.netloc.lower()


def load_trace_from_url(url: str, local_path: Optional[Path] = None) -> str:
    """
    Load trace from URL, handling both regular HTTP and SharePoint URLs.

    Args:
        url: URL to the trace file (HTTP or SharePoint)
        local_path: Optional local path or directory to save the file.
                   If a directory, filename is extracted from URL.

    Returns:
        str: Absolute path to the downloaded file

    Raises:
        ValueError: If URL scheme is not supported
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported URL scheme: {url}")

    # For SharePoint download.aspx URLs, don't pre-create the path
    # Let load_trace_from_sharepoint handle filename extraction
    if is_sharepoint_url(url) and "download.aspx" in url.lower():
        # Pass the directory as-is, the SharePoint loader will get the real filename
        return load_trace_from_sharepoint(url, local_path)

    # Handle directory path - extract filename from URL
    if local_path is not None:
        local_path = Path(local_path)
        if local_path.is_dir() or (not local_path.exists() and not local_path.suffix):
            # It's a directory or looks like one - extract filename from URL
            filename = Path(urlparse(url).path).name or "downloaded_trace"
            # Decode URL-encoded filename
            filename = unquote(filename)
            local_path = local_path / filename

    if is_sharepoint_url(url):
        return load_trace_from_sharepoint(url, local_path)
    else:
        # Regular HTTP download - could be implemented here or use requests
        import requests
        from tqdm import tqdm

        if local_path is None:
            filename = Path(urlparse(url).path).name or "downloaded_trace"
            filename = unquote(filename)
            local_path = Path(tempfile.mkdtemp()) / filename

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading from {url}")

        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            with open(local_path, "wb") as f, tqdm(
                desc=local_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                    size = f.write(chunk)
                    progress_bar.update(size)

        logger.info(f"Successfully downloaded to: {local_path}")
        return str(local_path.absolute())


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser(description="Download trace files from SharePoint")
    parser.add_argument("url", help="SharePoint URL to the trace file")
    parser.add_argument("-o", "--output", help="Output file path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    local_file = load_trace_from_url(args.url, output_path)
    print(f"\n✓ Downloaded to: {local_file}")
