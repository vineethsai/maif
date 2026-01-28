# MAIF Limitations

This document honestly describes what MAIF can't do and why.

## Compression Limitations

### Hierarchical Semantic Compression (HSC)

**Current Reality:**
- Achieves ~1.5x compression on embedding vectors
- Uses DBSCAN clustering + vector quantization + Huffman coding
- Claims of "2.5-4x compression, up to 10x maximum" are NOT proven

**Why it's limited:**
1. **DBSCAN parameter tuning is data-dependent**
   - `eps` and `min_samples` require manual tuning
   - No automatic parameter selection
   - Poor parameters lead to bad clustering

2. **8-bit quantization loses information**
   - Reduces each cluster center to 8-bit indices
   - Significant information loss for complex embeddings
   - Reconstruction fidelity ~95% at best

3. **Huffman coding reaches information-theoretic limits**
   - Can only compress as much as entropy in the data
   - If data is random, no compression possible
   - Most embeddings have high entropy

4. **No Product Quantization**
   - Current vector quantization is naive
   - Proper PQ would achieve 2-4x better results
   - Not implemented yet

**What will fix it:**
- Product Quantization with proper k-means training
- Adaptive parameter selection based on data distribution
- More sophisticated entropy coding
- **Estimated effort:** 3-4 weeks of development

---

## Embedding Limitations

### TF-IDF Only (No Neural Embeddings)

**Current Reality:**
- Only sklearn TF-IDF available
- No neural embedding models (BERT, Sentence-Transformers, etc.)
- 384-dimensional vectors from TF-IDF
- Good for keyword search, not semantic understanding

**Why it's limited:**
1. **TF-IDF is shallow**
   - Doesn't understand context or semantics
   - High sparsity (mostly zeros)
   - No understanding of word relationships

2. **Neural models are heavy dependencies**
   - Sentence-transformers = 300+ MB
   - PyTorch/TensorFlow = 1+ GB
   - Not suitable for lightweight deployments

3. **Model management is complex**
   - Multiple models to support
   - GPU vs CPU implications
   - Cache invalidation issues
   - Licensing and attribution

**What will fix it:**
- Optional neural embeddings with ONNX runtime
- Lightweight ONNX models (~50 MB)
- Model caching and versioning
- **Estimated effort:** 2 weeks

**What won't change:**
- MAIF will NOT require neural models by default
- Core features will always work with TF-IDF
- Neural embeddings will be optional add-on

---

## ACAM (Adaptive Cross-Modal Attention) Limitations

### Slow Training

**Current Reality:**
- Uses finite difference gradient descent
- O(n²) complexity per epoch for parameter updates
- Takes minutes to train on small datasets
- Not suitable for real-time learning

**Why it's slow:**
```python
def _compute_gradient(self, W, sample, epsilon=1e-5):
    gradient = np.zeros_like(W)  # Create full-size gradient
    for i in range(W.shape[0]):  # For every row
        for j in range(W.shape[1]):  # For every column
            # Compute finite difference (2 forward passes)
            W_plus[i,j] += epsilon
            loss_plus = self._compute_loss(sample)  # Full recompute
            loss_minus = ...
            gradient[i,j] = (loss_plus - loss_minus) / (2*epsilon)
```

For a 384x384 matrix:
- 147,456 gradient computations
- Each needs full loss recomputation
- Seconds per parameter = minutes per epoch

**What will fix it:**
- Switch to autodiff (JAX or PyTorch)
- Proper backpropagation algorithm
- Vectorized operations
- **Estimated effort:** 2 weeks

**What won't change:**
- ACAM training will never be super fast
- Some manual tuning will still be needed
- Not suitable for large-scale online learning

---

## Small Batch Performance

### Poor Performance with Small Batches

**Current Reality:**
- Works fine with 100+ samples
- Performance degrades with 10 samples
- Clustering algorithms need sufficient data
- Statistics become unreliable

**Why it's limited:**
1. **DBSCAN needs density**
   - Requires `min_samples` = len(embeddings) // 20
   - With 10 samples: min_samples = 0
   - Parameters become meaningless

2. **K-means struggles with few samples**
   - 256 clusters from 10 samples = 25x oversampling
   - Each cluster has <1 sample on average
   - Kmeans++ initialization fails

3. **Entropy coding needs frequency**
   - Huffman codes need repeated values
   - 10 unique samples = 10 unique values
   - No compression benefit

**What will fix it:**
- Better algorithm selection based on n_samples
- Different compression for small vs large batches
- Minimum sample requirements in API
- **Estimated effort:** 1 week

**Current workaround:**
```python
if len(embeddings) < 100:
    # Use standard compression instead
    use_zlib_compression()
else:
    # Use HSC
    use_hsc_compression()
```

---

## CSB (Cryptographic Semantic Binding) Limitations

### Not Validated for High-Security Use

**Current Reality:**
- Uses SHA-256 commitment schemes
- Implements Schnorr-like zero-knowledge proofs
- Infrastructure is sound but not audited
- Not tested against adversarial inputs

**Why it's limited:**
1. **No formal security proof**
   - Not peer-reviewed
   - No cryptographic audit
   - May have implementation vulnerabilities

2. **Schnorr-like proofs are simplified**
   - Not true Schnorr protocol
   - May have replay vulnerabilities
   - Challenge generation may be predictable

3. **No protection against quantum attacks**
   - SHA-256 is quantum-resistant
   - But discrete log schemes aren't
   - Actual ZK proofs use discrete log

**What will fix it:**
- Full cryptographic security audit
- Peer review by security researchers
- Formal verification of proofs
- **Estimated effort:** 4-6 weeks + external review

**Current status:**
- Safe for tamper detection
- NOT recommended for cryptographic verification
- NOT suitable for regulatory compliance alone

---

## Neural Embedder (❌ Not Implemented)

### What Doesn't Work

**Status: ❌ NOT FUNCTIONAL**

```python
from maif.semantic import NeuralEmbedder

# This will NOT work:
embedder = NeuralEmbedder("sentence-transformers/all-MiniLM-L6-v2")
# ❌ ImportError or AttributeError
```

**Why it's not implemented:**
1. Heavy dependencies conflict with lightweight design
2. Model downloading and caching is complex
3. GPU/CPU selection needs careful handling
4. License attribution requirements
5. Version compatibility issues

**Planned for v2.1:**
```python
from maif.semantic.neural import OptionalNeuralEmbedder

# New optional module
embedder = OptionalNeuralEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    cache_dir="~/.maif/models"
)
embeddings = embedder.embed_texts(texts)
```

**Timeline:** 2-3 weeks after v2.1 release

---

## What Won't Change (By Design)

### 1. MAIF is not a vector database
- It's a file format, not a search engine
- Use Chroma, Weaviate, or Pinecone for vector search
- MAIF stores embeddings, doesn't query them efficiently

### 2. MAIF is not a neural network library
- It's a container format
- Embeddings are pre-computed
- It doesn't train models

### 3. MAIF is not a general compression library
- Optimized for semantic data (embeddings, text)
- Not suitable for arbitrary binary data
- For general compression, use ZLIB/BROTLI

### 4. MAIF is not real-time ready
- File I/O has latency
- Suitable for batch operations
- Not for sub-millisecond operations

### 5. MAIF is not a distributed system
- Single file per MAIF artifact
- No built-in replication
- Use S3 or git for distribution

---

## Workarounds for Common Limitations

### Problem: Need neural embeddings now
**Solution:**
```python
from sentence_transformers import SentenceTransformer

# Embed outside MAIF, store vectors
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(texts)

# Add to MAIF
maif.add_embeddings(embeddings, model_name="sentence-transformers/...")
```

### Problem: HSC compression not good enough
**Solution:**
```python
# Use standard compression instead
maif.add_embedding_block(
    embeddings,
    compression="brotli",  # Better than HSC
    compression_level=11
)
```

### Problem: Need faster ACAM training
**Solution:**
```python
# Pre-train on larger dataset, load weights
acam = AdaptiveCrossModalAttention(...)
acam.load_weights("pretrained.pkl")

# Fine-tune or use as-is
weights = acam.compute_attention_weights(new_embeddings)
```

### Problem: Small batch performance
**Solution:**
```python
# Accumulate samples until you have 100+
buffer = []
for sample in stream:
    buffer.append(sample)
    if len(buffer) >= 100:
        # Process batch
        process_batch(buffer)
        buffer = []
```

---

## Contributing to Fixes

Want to help fix these limitations?

1. **Implement Product Quantization** (compression)
   - File: `maif/semantic/semantic_optimized.py`
   - Effort: 3-4 weeks
   - Impact: 2-4x better compression

2. **Add neural embedding support** (optional module)
   - File: `maif/semantic/neural.py` (new)
   - Effort: 2-3 weeks
   - Impact: Semantic search quality

3. **Optimize ACAM** (use autodiff)
   - File: `maif/semantic/acam_optimized.py` (new)
   - Effort: 2 weeks
   - Impact: 100x faster training

4. **Security audit of CSB**
   - Requires external review
   - Effort: 4-6 weeks
   - Impact: Cryptographic guarantees

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
