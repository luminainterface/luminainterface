import httpx
import asyncio
import numpy as np
from typing import List, Dict
import json
import time
from datetime import datetime

def create_drifted_embeddings(base_embeddings: np.ndarray, drift_type: str = 'systematic') -> np.ndarray:
    """Create drifted embeddings using different strategies"""
    if drift_type == 'systematic':
        # Create a systematic shift in a particular direction
        direction = np.random.randn(1, base_embeddings.shape[1])
        direction = direction / np.linalg.norm(direction)
        magnitude = 5.0  # Increased from 2.0
        
        # Apply directional shift and add noise
        drifted = base_embeddings + direction * magnitude
        drifted += np.random.randn(*base_embeddings.shape) * 1.0
        
    elif drift_type == 'expansion':
        # Make embeddings more spread out
        center = base_embeddings.mean(axis=0, keepdims=True)
        drifted = (base_embeddings - center) * 4.0  # Increased from 2.0
        drifted += center
        drifted += np.random.randn(*base_embeddings.shape) * 0.5
        
    elif drift_type == 'rotation':
        # Apply a random rotation matrix and scaling
        dim = base_embeddings.shape[1]
        random_matrix = np.random.randn(dim, dim)
        q, r = np.linalg.qr(random_matrix)
        drifted = base_embeddings @ q * 3.0  # Added scaling
        drifted += np.random.randn(*base_embeddings.shape) * 0.5
    
    # Normalize to maintain scale
    scale = np.linalg.norm(base_embeddings, axis=1, keepdims=True)
    drifted = drifted / np.linalg.norm(drifted, axis=1, keepdims=True) * scale * 2.0  # Double the scale
    
    print(f"\nDrift analysis for {drift_type}:")
    print(f"Original norm: {np.linalg.norm(base_embeddings):.4f}")
    print(f"Drifted norm: {np.linalg.norm(drifted):.4f}")
    print(f"Mean shift: {np.linalg.norm(np.mean(drifted, axis=0) - np.mean(base_embeddings, axis=0)):.4f}")
    print(f"Std change: {np.linalg.norm(np.std(drifted, axis=0) - np.std(base_embeddings, axis=0)):.4f}")
    
    return drifted

async def test_system_growth():
    """Test the system's growth functionality end-to-end"""
    async with httpx.AsyncClient(base_url="http://localhost:8905") as client:
        # Test multiple concepts with different drift types
        concepts = [
            ("quantum_mechanics", 0, "systematic"),
            ("neural_networks", 1, "expansion"),
            ("machine_learning", 2, "rotation")
        ]
        
        # Initial training phase
        print("\n=== Initial Training Phase ===")
        initial_embeddings = {}
        
        for concept_id, label, drift_type in concepts:
            # Generate initial embeddings
            embeddings = np.random.randn(20, 768)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            initial_embeddings[concept_id] = embeddings
            
            # Train concept
            response = await client.post(
                "/train",
                json={
                    "concept_id": concept_id,
                    "embeddings": embeddings.tolist(),
                    "labels": [label] * 20
                }
            )
            assert response.status_code == 200
            result = response.json()
            print(f"\nTrained {concept_id}:")
            print(json.dumps(result, indent=2))
            
            # Get initial metrics
            response = await client.get(f"/concepts/{concept_id}/metrics")
            assert response.status_code == 200
            metrics = response.json()
            print(f"\nInitial metrics for {concept_id}:")
            print(json.dumps(metrics, indent=2))
        
        # Get initial network stats
        response = await client.get("/network/stats")
        assert response.status_code == 200
        initial_stats = response.json()
        print("\nInitial network stats:")
        print(json.dumps(initial_stats, indent=2))
        
        initial_growth_events = initial_stats["growth_events"]
        
        # Simulate concept drift and growth phase
        print("\n=== Drift and Growth Phase ===")
        for concept_id, label, drift_type in concepts:
            print(f"\nProcessing {concept_id} with {drift_type} drift...")
            
            # Generate drifted embeddings using different strategies
            base_embeddings = initial_embeddings[concept_id]
            drifted_embeddings = create_drifted_embeddings(base_embeddings, drift_type)
            
            # Train with drifted data
            response = await client.post(
                "/train",
                json={
                    "concept_id": concept_id,
                    "embeddings": drifted_embeddings.tolist(),
                    "labels": [label] * 20
                }
            )
            assert response.status_code == 200
            result = response.json()
            print(f"\nTrained {concept_id} with {drift_type} drift:")
            print(json.dumps(result, indent=2))
            
            # Check if growth is needed
            if result["drift"] > 0.001:  # Using new threshold
                print(f"\nDrift {result['drift']} exceeds threshold 0.001, attempting growth...")
                
                # Find the smallest layer to grow
                layer_sizes = initial_stats["layer_sizes"]
                layer_to_grow = layer_sizes.index(min(layer_sizes))
                current_size = layer_sizes[layer_to_grow]
                new_size = int(current_size * 1.5)  # Grow by 50%
                
                print(f"Growing layer {layer_to_grow} from {current_size} to {new_size}...")
                
                response = await client.post(
                    "/grow",
                    json={
                        "concept_id": concept_id,
                        "layer_idx": layer_to_grow,
                        "new_size": new_size
                    }
                )
                assert response.status_code == 200
                growth_result = response.json()
                print(f"Growth result:")
                print(json.dumps(growth_result, indent=2))
                
                # Wait for growth to take effect
                time.sleep(1)
                
                # Get updated metrics
                response = await client.get(f"/concepts/{concept_id}/metrics")
                assert response.status_code == 200
                metrics = response.json()
                print(f"\nUpdated metrics for {concept_id}:")
                print(json.dumps(metrics, indent=2))
            else:
                print(f"\nDrift {result['drift']} below threshold 0.001, skipping growth")
        
        # Get final network stats
        response = await client.get("/network/stats")
        assert response.status_code == 200
        final_stats = response.json()
        print("\nFinal network stats:")
        print(json.dumps(final_stats, indent=2))
        
        # Verify growth occurred
        assert final_stats["growth_events"] > initial_growth_events, "No new growth events occurred"
        
        # Check if any layer grew
        layers_grew = False
        for i in range(len(initial_stats["layer_sizes"])):
            if final_stats["layer_sizes"][i] > initial_stats["layer_sizes"][i]:
                layers_grew = True
                break
        assert layers_grew, "No layers grew in size"
        
        print("\n=== Growth Verification ===")
        print(f"Initial layer sizes: {initial_stats['layer_sizes']}")
        print(f"Final layer sizes: {final_stats['layer_sizes']}")
        print(f"Initial growth events: {initial_growth_events}")
        print(f"Final growth events: {final_stats['growth_events']}")
        print(f"Concepts tracked: {final_stats['concepts_tracked']}")

if __name__ == "__main__":
    asyncio.run(test_system_growth()) 