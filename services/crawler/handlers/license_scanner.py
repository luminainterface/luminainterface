#!/usr/bin/env python3
"""License scanner for repository validation.

This module handles:
1. Repository cloning and license scanning
2. License validation against allowed list
3. Robots.txt checking
4. Metrics for license blocks and robots blocks
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import httpx
from prometheus_client import Counter
from robotexclusionrulesparser import RobotExclusionRulesParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('license-scanner')

# Prometheus metrics
LICENSE_BLOCKS = Counter('crawler_license_block_total', 'Total number of license blocks')
ROBOTS_BLOCKS = Counter('crawler_robots_block_total', 'Total number of robots.txt blocks')

# Allowed licenses (SPDX identifiers)
ALLOWED_LICENSES = {
    'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC',
    'Unlicense', 'CC0-1.0', 'CC-BY-4.0', 'CC-BY-SA-4.0'
}

class LicenseScanner:
    def __init__(self, scancode_path: str = 'scancode'):
        """Initialize the license scanner.
        
        Args:
            scancode_path: Path to scancode executable
        """
        self.scancode_path = scancode_path
        self.http = httpx.AsyncClient(timeout=30.0)
        self.robots_parser = RobotExclusionRulesParser()
        
    async def check_repo(self, repo_url: str) -> Tuple[bool, Optional[str]]:
        """Check a repository for license and robots.txt compliance.
        
        Args:
            repo_url: URL of the repository to check
            
        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        # First check robots.txt
        try:
            host = httpx.URL(repo_url).host
            robots_url = f"https://{host}/robots.txt"
            robots_resp = await self.http.get(robots_url)
            if robots_resp.status_code == 200:
                self.robots_parser.parse(robots_resp.text)
                if not self.robots_parser.is_allowed("*", repo_url):
                    ROBOTS_BLOCKS.inc()
                    return False, "Blocked by robots.txt"
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {repo_url}: {e}")
            
        # Clone and scan for license
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Clone repo
                clone_cmd = [
                    'git', 'clone', '--filter=blob:none', '--depth', '1',
                    repo_url, tmpdir
                ]
                result = subprocess.run(
                    clone_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Run scancode
                scan_cmd = [
                    self.scancode_path,
                    '--license',
                    '--json', f'{tmpdir}/scan.json',
                    tmpdir
                ]
                result = subprocess.run(
                    scan_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Parse results
                scan_path = Path(tmpdir) / 'scan.json'
                if not scan_path.exists():
                    logger.warning(f"No scan results for {repo_url}")
                    return False, "No license found"
                    
                with open(scan_path) as f:
                    import json
                    scan_data = json.load(f)
                    
                # Check licenses
                found_licenses = set()
                for file_info in scan_data.get('files', []):
                    for license_info in file_info.get('licenses', []):
                        spdx = license_info.get('spdx_license_key')
                        if spdx:
                            found_licenses.add(spdx)
                            
                if not found_licenses:
                    LICENSE_BLOCKS.inc()
                    return False, "No license found"
                    
                # Check if any license is allowed
                if not any(lic in ALLOWED_LICENSES for lic in found_licenses):
                    LICENSE_BLOCKS.inc()
                    return False, f"License not allowed: {', '.join(found_licenses)}"
                    
                return True, None
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error scanning {repo_url}: {e.stderr}")
                return False, f"Scan failed: {e.stderr}"
            except Exception as e:
                logger.error(f"Error processing {repo_url}: {e}")
                return False, str(e)
                
    async def close(self):
        """Cleanup connections."""
        await self.http.aclose() 