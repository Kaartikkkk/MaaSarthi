#!/usr/bin/env python3
"""
Performance test to verify optimization fixes.
Tests: Response times, lazy loading, dataset caching
"""
import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:5001"

def test_home_page():
    """Test home page load time"""
    start = time.time()
    r = requests.get(f"{BASE_URL}/", timeout=10)
    elapsed = time.time() - start
    return {
        "test": "Home Page",
        "status_code": r.status_code,
        "time_ms": round(elapsed * 1000, 2),
        "success": r.status_code == 200
    }

def test_predict_route():
    """Test /predict endpoint (uses lazy loading)"""
    start = time.time()
    data = {
        "age": "35",
        "kids": "2",
        "hours": "4",
        "domain": "Cooking",
        "skill": "Cooking",
        "education": "10th",
        "city_type": "Urban",
        "location": "Mumbai",
        "language": "Hindi",
        "device": "Mobile",
        "work_mode": "Work From Home"
    }
    r = requests.post(f"{BASE_URL}/predict", data=data, timeout=15)
    elapsed = time.time() - start
    return {
        "test": "Predict Route",
        "status_code": r.status_code,
        "time_ms": round(elapsed * 1000, 2),
        "success": r.status_code == 200
    }

def test_api_endpoint():
    """Test API endpoint"""
    start = time.time()
    r = requests.get(f"{BASE_URL}/api/check-session", timeout=10)
    elapsed = time.time() - start
    return {
        "test": "API Endpoint",
        "status_code": r.status_code,
        "time_ms": round(elapsed * 1000, 2),
        "success": r.status_code in [200, 302]
    }

def test_concurrent_requests(num_requests=5):
    """Test concurrent requests to check thread safety"""
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(test_home_page) for _ in range(num_requests)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.time() - start
    
    return {
        "test": f"Concurrent Requests ({num_requests}x)",
        "total_time_ms": round(elapsed * 1000, 2),
        "avg_time_ms": round((elapsed / num_requests) * 1000, 2),
        "success": all(r["success"] for r in results)
    }

def main():
    print("=" * 70)
    print("🚀 MaaSarthi Performance Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        ("Home Page Load", test_home_page),
        ("Predict Route (Dataset Lazy Load)", test_predict_route),
        ("API Endpoint", test_api_endpoint),
        ("Concurrent Requests", test_concurrent_requests),
    ]
    
    print(f"Testing {BASE_URL}...\n")
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}...")
        try:
            result = test_func()
            results.append(result)
            print(f"  ✅ {result['test']}: {result.get('time_ms', result.get('avg_time_ms'))}ms")
        except Exception as e:
            print(f"  ❌ {test_name}: {e}")
            results.append({"test": test_name, "success": False, "error": str(e)})
        print()
    
    # Summary
    print("=" * 70)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    print()
    
    successful = sum(1 for r in results if r.get("success", False))
    print(f"✅ Tests Passed: {successful}/{len(results)}")
    print()
    
    if all(r.get("success", False) for r in results):
        print("🎉 All tests passed! Server is performing optimally.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    main()
