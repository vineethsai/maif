# MAIF Benchmark Suite

This benchmark suite validates the key claims made in the research paper "An Artifact-Centric AI Agent Design and the Multimodal Artifact File Format (MAIF) for Enhanced Trustworthiness".

## Overview

The benchmark suite tests the following paper claims:

### Performance Claims
- **2.5-5× compression ratios** for text data
- **Sub-50ms semantic search** on commodity hardware  
- **500+ MB/s streaming throughput** for large files
- **<15% cryptographic overhead** for security features
- **95%+ automated repair success rates** for corrupted files

### Security Claims
- **100% tamper detection** within 1ms verification
- **Immutable provenance chains** with cryptographic linking
- **Block-level integrity verification** with hash validation

### Functionality Claims
- **Multimodal data integration** (text, binary, embeddings)
- **Semantic embedding and search** capabilities
- **Privacy-by-design features** with encryption and anonymization
- **Cross-modal attention mechanisms** for AI processing

## Running the Benchmarks

### Quick Start

```bash
# Run the complete benchmark suite
python run_benchmark.py

# Run with reduced test sizes (faster)
python run_benchmark.py --quick
```

### Direct Execution

```bash
# Run the benchmark module directly
python -m benchmarks.maif_benchmark_suite

# Specify custom output directory
python -m benchmarks.maif_benchmark_suite --output-dir ./my_results
```

### Requirements

The benchmark requires the following Python packages:

```bash
pip install numpy
pip install sentence-transformers  # Optional, for semantic embeddings
pip install cryptography           # For security features
pip install brotli                # Optional, for advanced compression
pip install lz4                   # Optional, for LZ4 compression
pip install zstandard             # Optional, for Zstandard compression
```

## Benchmark Components

### 1. Compression Ratio Tests
- Tests various compression algorithms (zlib, Brotli, LZMA)
- Measures compression ratios on different text types
- Validates the 2.5-5× compression claim

### 2. Semantic Search Performance
- Creates test corpus with 1000+ documents
- Measures search latency across multiple queries
- Validates sub-50ms search claim

### 3. Streaming Throughput
- Tests sequential and parallel block streaming
- Measures throughput on large files (10+ MB)
- Validates 500+ MB/s throughput claim

### 4. Cryptographic Overhead
- Compares encoding times with/without encryption
- Measures performance impact of security features
- Validates <15% overhead claim

### 5. Tamper Detection
- Introduces controlled file corruption
- Measures detection accuracy and speed
- Validates 100% detection within 1ms claim

### 6. Repair Capabilities
- Tests automated repair on corrupted files
- Measures repair success rates
- Validates 95%+ success rate claim

### 7. Multimodal Integration
- Tests storage and retrieval of different data types
- Validates cross-modal functionality
- Tests semantic relationships

### 8. Provenance Chains
- Tests cryptographic provenance linking
- Validates chain integrity
- Tests immutability guarantees

### 9. Privacy Features
- Tests encryption/decryption functionality
- Tests data anonymization
- Validates privacy-by-design claims

### 10. Scalability Tests
- Tests performance with large numbers of blocks
- Measures memory usage and processing time
- Validates scalability characteristics

## Output and Results

### Benchmark Report

The benchmark generates a comprehensive JSON report containing:

```json
{
  "timestamp": "...",
  "total_benchmarks": 10,
  "successful_benchmarks": 9,
  "failed_benchmarks": 1,
  "results": {
    "Compression Ratios": {
      "success": true,
      "metrics": {
        "average_ratio": 3.2,
        "claim_validation": {
          "paper_claim": "2.5-5× compression ratios",
          "achieved_avg": 3.2,
          "meets_claim": true
        }
      }
    }
  },
  "paper_claims_validation": {
    "claims_met": 8,
    "total_claims": 10,
    "claims_percentage": 80.0
  },
  "overall_assessment": {
    "implementation_maturity": "Beta Quality"
  }
}
```

### Maturity Assessment

The benchmark classifies implementation maturity as:

- **Production Ready**: ≥90% benchmarks pass, ≥80% claims validated
- **Beta Quality**: ≥70% benchmarks pass, ≥60% claims validated  
- **Alpha Quality**: ≥50% benchmarks pass, ≥40% claims validated
- **Prototype**: <50% benchmarks pass or <40% claims validated

### Console Output

The benchmark provides real-time progress updates:

```
MAIF BENCHMARK SUITE - VALIDATING PAPER CLAIMS
================================================================================
 Compression Ratios: Avg 3.2×
 Semantic Search: Avg 35.2ms
 Streaming Throughput: 650.1 MB/s
 Cryptographic Overhead: 12.3%
 Tamper Detection: 100.0% in 0.85ms
 Integrity Verification: 245.3 MB/s
 Multimodal Integration: 3 blocks
 Provenance Chains: 100 entries
 Privacy Features: Encryption & Anonymization
 Repair Capabilities: 96.0% success
 Scalability: Up to 10000 blocks

Paper Claims Validation: 9/10 (90.0%)
Overall Implementation Status: Production Ready
```

## Interpreting Results

### Success Criteria

A benchmark is considered successful if:
1. It executes without errors
2. It meets or exceeds the paper's claimed performance
3. It demonstrates the claimed functionality

### Common Issues

- **Missing Dependencies**: Install optional packages for full functionality
- **Hardware Limitations**: Some performance claims require modern hardware
- **Memory Constraints**: Large-scale tests may require sufficient RAM

### Troubleshooting

If benchmarks fail:

1. Check that all required dependencies are installed
2. Ensure sufficient disk space for temporary files
3. Verify that the system has adequate memory
4. Run with `--quick` flag for reduced resource usage

## Extending the Benchmark

### Adding New Tests

To add a new benchmark:

1. Create a method in `MAIFBenchmarkSuite` class
2. Follow the naming pattern `_benchmark_<feature_name>`
3. Use `BenchmarkResult` to track metrics
4. Add claim validation logic
5. Update the `run_all_benchmarks()` method

### Custom Metrics

Add custom metrics using:

```python
result.add_metric("custom_metric", value)
result.add_metric("claim_validation", {
    "paper_claim": "Description of claim",
    "achieved": actual_value,
    "meets_claim": actual_value >= expected_value
})
```

## Contributing

To contribute to the benchmark suite:

1. Fork the repository
2. Add new benchmark tests
3. Update documentation
4. Submit a pull request

## License

This benchmark suite is released under the same license as the MAIF project.