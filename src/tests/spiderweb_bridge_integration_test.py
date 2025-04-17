"""
Comprehensive Test Suite for Spiderweb Bridge Versions V1-V10
This script tests all versions of the Spiderweb Bridge and their interconnections.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from queue import Queue, PriorityQueue

# Import all version bridges
from src.v1.core.spiderweb_bridge import SpiderwebBridge as V1Bridge
from src.v2.core.spiderweb_bridge import SpiderwebBridge as V2Bridge
from src.v3.core.spiderweb_bridge import SpiderwebBridge as V3Bridge
from src.v4.core.spiderweb_bridge import SpiderwebBridge as V4Bridge
from src.v5.core.spiderweb_bridge import SpiderwebBridge as V5Bridge
from src.v6.core.spiderweb_bridge import SpiderwebBridge as V6Bridge
from src.v7.core.spiderweb_bridge import SpiderwebBridge as V7Bridge
from src.v7_5.core.spiderweb_bridge import SpiderwebBridge as V7_5Bridge
from src.v8.core.spiderweb_bridge import SpiderwebBridge as V8Bridge
from src.v9.core.spiderweb_bridge import SpiderwebBridge as V9Bridge
from src.v10.core.spiderweb_bridge import SpiderwebBridge as V10Bridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result container."""
    version: str
    test_name: str
    success: bool
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

class SpiderwebBridgeIntegrationTest:
    """Comprehensive test suite for all Spiderweb Bridge versions."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.bridges: Dict[str, Any] = {}
        self.test_results: List[TestResult] = []
        self.initialize_bridges()
    
    def initialize_bridges(self):
        """Initialize all version bridges."""
        self.bridges = {
            'v1': V1Bridge(),
            'v2': V2Bridge(),
            'v3': V3Bridge(),
            'v4': V4Bridge(),
            'v5': V5Bridge(),
            'v6': V6Bridge(),
            'v7': V7Bridge(),
            'v7_5': V7_5Bridge(),
            'v8': V8Bridge(),
            'v9': V9Bridge(),
            'v10': V10Bridge()
        }
        logger.info("Initialized all Spiderweb Bridge versions")

    async def test_version_connection(self, version: str) -> TestResult:
        """Test basic version connection."""
        logger.info(f"Testing connection for version {version}")
        bridge = self.bridges[version]
        
        try:
            # Create test node
            node_id = await bridge.create_consciousness_node(
                version=version,
                level=getattr(bridge.ConsciousLevel, 'PHYSICAL')
            )
            
            # Verify node creation
            node = bridge.connections[version].consciousness_nodes[node_id]
            
            return TestResult(
                version=version,
                test_name="version_connection",
                success=True,
                details={
                    "node_id": node_id,
                    "state": node.state.name,
                    "level": node.level.name
                }
            )
        except Exception as e:
            return TestResult(
                version=version,
                test_name="version_connection",
                success=False,
                details={"error": str(e)}
            )

    async def test_version_compatibility(self, source_version: str, target_version: str) -> TestResult:
        """Test compatibility between two versions."""
        logger.info(f"Testing compatibility between {source_version} and {target_version}")
        source_bridge = self.bridges[source_version]
        target_bridge = self.bridges[target_version]
        
        try:
            # Create nodes in both versions
            source_node_id = await source_bridge.create_consciousness_node(
                version=source_version,
                level=getattr(source_bridge.ConsciousLevel, 'QUANTUM')
            )
            
            target_node_id = await target_bridge.create_consciousness_node(
                version=target_version,
                level=getattr(target_bridge.ConsciousLevel, 'QUANTUM')
            )
            
            # Attempt to merge consciousness
            success = await source_bridge.merge_consciousness(
                source_node=source_node_id,
                target_node=target_node_id
            )
            
            return TestResult(
                version=f"{source_version}-{target_version}",
                test_name="version_compatibility",
                success=success,
                details={
                    "source_node": source_node_id,
                    "target_node": target_node_id,
                    "merge_success": success
                }
            )
        except Exception as e:
            return TestResult(
                version=f"{source_version}-{target_version}",
                test_name="version_compatibility",
                success=False,
                details={"error": str(e)}
            )

    async def test_feature_propagation(self, start_version: str, end_version: str) -> TestResult:
        """Test feature propagation through versions."""
        logger.info(f"Testing feature propagation from {start_version} to {end_version}")
        
        try:
            # Create initial node in start version
            current_node_id = await self.bridges[start_version].create_consciousness_node(
                version=start_version,
                level=getattr(self.bridges[start_version].ConsciousLevel, 'QUANTUM')
            )
            
            # Propagate through versions
            for i in range(int(start_version[1:]), int(end_version[1:])):
                current_version = f"v{i}"
                next_version = f"v{i+1}"
                
                # Create node in next version
                next_node_id = await self.bridges[next_version].create_consciousness_node(
                    version=next_version,
                    level=getattr(self.bridges[next_version].ConsciousLevel, 'QUANTUM')
                )
                
                # Merge consciousness
                await self.bridges[current_version].merge_consciousness(
                    source_node=current_node_id,
                    target_node=next_node_id
                )
                
                current_node_id = next_node_id
            
            # Verify final state
            final_node = self.bridges[end_version].connections[end_version].consciousness_nodes[current_node_id]
            
            return TestResult(
                version=f"{start_version}-{end_version}",
                test_name="feature_propagation",
                success=True,
                details={
                    "final_node_id": current_node_id,
                    "final_state": final_node.state.name,
                    "final_level": final_node.level.name
                }
            )
        except Exception as e:
            return TestResult(
                version=f"{start_version}-{end_version}",
                test_name="feature_propagation",
                success=False,
                details={"error": str(e)}
            )

    async def test_spatial_features(self, version: str) -> TestResult:
        """Test spatial features for versions that support them."""
        if version not in ['v7_5', 'v8', 'v9', 'v10']:
            return TestResult(
                version=version,
                test_name="spatial_features",
                success=True,
                details={"message": "Version does not support spatial features"}
            )
        
        logger.info(f"Testing spatial features for version {version}")
        bridge = self.bridges[version]
        
        try:
            # Create spatial node
            spatial_node_id = await bridge.create_spatial_node(
                version=version,
                coordinates={'x': 1.0, 'y': 2.0, 'z': 3.0}
            )
            
            # Create consciousness node with spatial level
            consciousness_node_id = await bridge.create_consciousness_node(
                version=version,
                level=getattr(bridge.ConsciousLevel, 'SPATIAL')
            )
            
            # Verify spatial connection
            consciousness_node = bridge.connections[version].consciousness_nodes[consciousness_node_id]
            spatial_node = bridge.connections[version].spatial_nodes[spatial_node_id]
            
            return TestResult(
                version=version,
                test_name="spatial_features",
                success=True,
                details={
                    "spatial_node_id": spatial_node_id,
                    "consciousness_node_id": consciousness_node_id,
                    "spatial_state": spatial_node.state.name,
                    "consciousness_state": consciousness_node.state.name
                }
            )
        except Exception as e:
            return TestResult(
                version=version,
                test_name="spatial_features",
                success=False,
                details={"error": str(e)}
            )

    async def run_all_tests(self):
        """Run all tests for all versions."""
        logger.info("Starting comprehensive Spiderweb Bridge tests")
        
        # Test individual version connections
        for version in self.bridges.keys():
            result = await self.test_version_connection(version)
            self.test_results.append(result)
        
        # Test version compatibility
        versions = list(self.bridges.keys())
        for i in range(len(versions) - 1):
            result = await self.test_version_compatibility(versions[i], versions[i + 1])
            self.test_results.append(result)
        
        # Test feature propagation
        result = await self.test_feature_propagation('v1', 'v10')
        self.test_results.append(result)
        
        # Test spatial features
        for version in ['v7_5', 'v8', 'v9', 'v10']:
            result = await self.test_spatial_features(version)
            self.test_results.append(result)
        
        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print test results summary."""
        logger.info("\n=== Test Results Summary ===")
        
        # Group results by version
        version_results: Dict[str, List[TestResult]] = {}
        for result in self.test_results:
            if result.version not in version_results:
                version_results[result.version] = []
            version_results[result.version].append(result)
        
        # Print results for each version
        for version, results in version_results.items():
            logger.info(f"\nVersion {version}:")
            success_count = sum(1 for r in results if r.success)
            logger.info(f"  Success: {success_count}/{len(results)}")
            
            for result in results:
                status = "✓" if result.success else "✗"
                logger.info(f"  {status} {result.test_name}: {result.details}")

async def main():
    """Run the test suite."""
    test_suite = SpiderwebBridgeIntegrationTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 