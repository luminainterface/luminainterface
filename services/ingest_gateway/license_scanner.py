import os
import json
import logging
import tempfile
import subprocess
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge

# Metrics
LICENSE_SCAN_TOTAL = Counter(
    'license_scan_total',
    'Total number of license scans performed',
    ['result']  # 'allowed', 'blocked', 'error'
)

LICENSE_SCAN_DURATION = Histogram(
    'license_scan_duration_seconds',
    'Time spent scanning for licenses'
)

LICENSE_DISTRIBUTION = Gauge(
    'license_distribution',
    'Distribution of detected licenses',
    ['license_type']
)

@dataclass
class LicenseInfo:
    """Information about detected licenses."""
    spdx_key: str
    name: str
    confidence: float
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    matched_text: Optional[str] = None
    is_deprecated: bool = False
    is_blocked: bool = False

@dataclass
class ScanResult:
    """Result of a license scan."""
    licenses: List[LicenseInfo]
    is_blocked: bool
    scan_time: datetime
    error: Optional[str] = None
    warning: Optional[str] = None

class LicenseScanner:
    """Enhanced license scanner using scancode-toolkit."""
    
    # License categories
    BLOCKED_LICENSES = {
        "GPL-1.0", "GPL-2.0", "GPL-3.0", "AGPL-1.0", "AGPL-3.0",
        "LGPL-2.0", "LGPL-2.1", "LGPL-3.0", "CC-BY-SA-1.0",
        "CC-BY-SA-2.0", "CC-BY-SA-3.0", "CC-BY-SA-4.0"
    }
    
    DEPRECATED_LICENSES = {
        "GPL-1.0", "AGPL-1.0", "LGPL-2.0", "CC-BY-SA-1.0",
        "CC-BY-SA-2.0"
    }
    
    ALLOWED_LICENSES = {
        "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause",
        "ISC", "CC0-1.0", "CC-BY-4.0", "Unlicense"
    }
    
    def __init__(self, min_confidence: float = 0.8):
        self.min_confidence = min_confidence
        self.logger = logging.getLogger("license_scanner")
        
        # Verify scancode installation
        try:
            subprocess.run(["scancode", "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error("scancode-toolkit not found. Please install it first.")
            raise RuntimeError("scancode-toolkit not found")
    
    def _parse_scan_output(self, scan_data: Dict) -> List[LicenseInfo]:
        """Parse scancode output into LicenseInfo objects."""
        licenses = []
        
        for license_info in scan_data.get("licenses", []):
            spdx_key = license_info.get("spdx_license_key")
            if not spdx_key:
                continue
                
            confidence = float(license_info.get("score", 0))
            if confidence < self.min_confidence:
                continue
            
            licenses.append(LicenseInfo(
                spdx_key=spdx_key,
                name=license_info.get("short_name", spdx_key),
                confidence=confidence,
                start_line=license_info.get("start_line"),
                end_line=license_info.get("end_line"),
                matched_text=license_info.get("matched_text"),
                is_deprecated=spdx_key in self.DEPRECATED_LICENSES,
                is_blocked=spdx_key in self.BLOCKED_LICENSES
            ))
            
            # Update metrics
            LICENSE_DISTRIBUTION.labels(license_type=spdx_key).inc()
        
        return licenses
    
    async def scan_file(self, file_path: str) -> ScanResult:
        """Scan a file for licenses."""
        start_time = datetime.utcnow()
        
        try:
            # Run scancode with detailed output
            result = subprocess.run(
                [
                    "scancode",
                    "--json", "-",
                    "--license",
                    "--license-text",
                    "--no-progress",
                    file_path
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            scan_data = json.loads(result.stdout)
            licenses = self._parse_scan_output(scan_data)
            
            # Check for blocked licenses
            is_blocked = any(license.is_blocked for license in licenses)
            
            # Check for deprecated licenses
            deprecated = [l for l in licenses if l.is_deprecated]
            warning = None
            if deprecated:
                warning = f"Found deprecated licenses: {', '.join(l.spdx_key for l in deprecated)}"
            
            scan_result = ScanResult(
                licenses=licenses,
                is_blocked=is_blocked,
                scan_time=datetime.utcnow(),
                warning=warning
            )
            
            # Update metrics
            LICENSE_SCAN_TOTAL.labels(
                result="blocked" if is_blocked else "allowed"
            ).inc()
            
            LICENSE_SCAN_DURATION.observe(
                (datetime.utcnow() - start_time).total_seconds()
            )
            
            return scan_result
            
        except subprocess.CalledProcessError as e:
            error_msg = f"License scan failed: {e.stderr}"
            self.logger.error(error_msg)
            LICENSE_SCAN_TOTAL.labels(result="error").inc()
            return ScanResult(
                licenses=[],
                is_blocked=True,  # Block on error to be safe
                scan_time=datetime.utcnow(),
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error during license scan: {str(e)}"
            self.logger.error(error_msg)
            LICENSE_SCAN_TOTAL.labels(result="error").inc()
            return ScanResult(
                licenses=[],
                is_blocked=True,
                scan_time=datetime.utcnow(),
                error=error_msg
            )
    
    async def scan_content(self, content: bytes, file_extension: str = ".txt") -> ScanResult:
        """Scan content in memory for licenses."""
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            try:
                tmp.write(content)
                tmp_path = tmp.name
                return await self.scan_file(tmp_path)
            finally:
                os.unlink(tmp_path)
    
    def get_license_summary(self, scan_result: ScanResult) -> Dict:
        """Get a summary of license scan results."""
        return {
            "is_blocked": scan_result.is_blocked,
            "license_count": len(scan_result.licenses),
            "licenses": [
                {
                    "spdx_key": l.spdx_key,
                    "name": l.name,
                    "confidence": l.confidence,
                    "is_deprecated": l.is_deprecated,
                    "is_blocked": l.is_blocked
                }
                for l in scan_result.licenses
            ],
            "warning": scan_result.warning,
            "error": scan_result.error,
            "scan_time": scan_result.scan_time.isoformat()
        } 
 