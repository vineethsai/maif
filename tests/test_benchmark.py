#!/usr/bin/env python3
"""
Quick test to verify the benchmark suite works correctly.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_basic_maif_functionality():
    """Test basic MAIF functionality before running full benchmark."""
    print("Testing basic MAIF functionality...")

    try:
        from maif import MAIFEncoder, MAIFDecoder, BlockType

        with tempfile.TemporaryDirectory() as tmpdir:
            maif_path = os.path.join(tmpdir, "test.maif")

            # Test encoding (v3 format)
            encoder = MAIFEncoder(maif_path, agent_id="benchmark_test")
            encoder.add_text_block("Hello, MAIF!")
            encoder.add_embeddings_block([[1.0, 2.0, 3.0]])
            encoder.finalize()

            # Test decoding
            decoder = MAIFDecoder(maif_path)
            decoder.load()

            # Get blocks
            text_blocks = [
                b for b in decoder.blocks if b.header.block_type == BlockType.TEXT
            ]
            emb_blocks = [
                b for b in decoder.blocks if b.header.block_type == BlockType.EMBEDDINGS
            ]

            # Verify
            assert len(text_blocks) == 1
            assert text_blocks[0].data.decode("utf-8") == "Hello, MAIF!"
            assert len(emb_blocks) == 1

            print("Basic MAIF functionality works")

    except Exception as e:
        print(f"Basic MAIF test failed: {e}")
        assert False, f"Basic MAIF test failed: {e}"


def test_benchmark_imports():
    """Test that benchmark imports work."""
    import sys
    from pathlib import Path

    # Add benchmarks directory to path
    benchmark_dir = Path(__file__).parent.parent / "benchmarks"
    sys.path.insert(0, str(benchmark_dir))

    # Test that core MAIF imports used by benchmarks work
    from maif.core import MAIFEncoder, MAIFDecoder
    from maif.compression import MAIFCompressor, CompressionAlgorithm
    from maif.privacy import PrivacyEngine

    assert MAIFEncoder is not None
    assert MAIFDecoder is not None
    assert MAIFCompressor is not None
    assert PrivacyEngine is not None


def test_quick_benchmark():
    """Run a very quick benchmark test."""
    import tempfile
    import time
    from maif import MAIFEncoder, MAIFDecoder
    from maif.compression import MAIFCompressor, CompressionAlgorithm

    with tempfile.TemporaryDirectory() as tmpdir:
        maif_path = os.path.join(tmpdir, "benchmark.maif")

        # Benchmark encoding
        start = time.time()
        encoder = MAIFEncoder(maif_path, agent_id="benchmark")
        for i in range(10):
            encoder.add_text_block(f"Benchmark content block {i} " * 100)
        encoder.finalize()
        encode_time = time.time() - start

        # Benchmark decoding
        start = time.time()
        decoder = MAIFDecoder(maif_path)
        decoder.load()
        decode_time = time.time() - start

        # Benchmark compression
        compressor = MAIFCompressor()
        test_data = b"Benchmark test data " * 1000
        start = time.time()
        compressed = compressor.compress(test_data, CompressionAlgorithm.ZLIB)
        compress_time = time.time() - start

        # Verify reasonable performance (encoding 10 blocks in under 2 seconds)
        assert encode_time < 2.0, f"Encoding too slow: {encode_time}s"
        assert decode_time < 1.0, f"Decoding too slow: {decode_time}s"
        assert compress_time < 0.1, f"Compression too slow: {compress_time}s"
        assert len(decoder.blocks) == 10


def main():
    """Run all tests."""
    print("MAIF Benchmark Test Suite")
    print("=" * 40)

    tests = [
        test_basic_maif_functionality,
        test_benchmark_imports,
        test_quick_benchmark,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("All tests passed! The benchmark suite is ready to run.")
        print("\nTo run the full benchmark suite:")
        print("python run_benchmark.py")
        print("\nTo run a quick benchmark:")
        print("python run_benchmark.py --quick")
        return 0
    else:
        print("Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
