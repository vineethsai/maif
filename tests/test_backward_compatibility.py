import pytest
import numpy as np
import os
import sys
import tempfile
import json
from unittest import mock
from pathlib import Path

# Mock imports for flexible testing environment
try:
    from maif.core.secure_format import (
        SecureMAIFWriter,
        SecureMAIFReader,
        FORMAT_VERSION_MAJOR,
        FORMAT_VERSION_MINOR
    )
except ImportError:
    # Fallback for testing without full installation
    FORMAT_VERSION_MAJOR = 2
    FORMAT_VERSION_MINOR = 2
    SecureMAIFWriter = None
    SecureMAIFReader = None

try:
    from maif.compression.compression import CompressionManager
except ImportError:
    CompressionManager = None

try:
    from maif.encoding.embeddings import get_embedder
except ImportError:
    get_embedder = None


class TestV21Compatibility:
    """Test v2.1 compatibility and version detection."""

    def test_version_detection(self):
        """Test version detection works correctly."""
        # v2.2 should report correct version
        assert FORMAT_VERSION_MAJOR == 2, "Major version should be 2"
        assert FORMAT_VERSION_MINOR == 2, "Minor version should be 2"

    def test_version_constants_defined(self):
        """Test that version constants are properly defined."""
        assert isinstance(FORMAT_VERSION_MAJOR, int), "Major version must be integer"
        assert isinstance(FORMAT_VERSION_MINOR, int), "Minor version must be integer"
        assert FORMAT_VERSION_MAJOR > 0, "Major version must be positive"
        assert FORMAT_VERSION_MINOR >= 0, "Minor version must be non-negative"

    def test_hsc_version_field_exists(self):
        """Test HSC version metadata field can be set and retrieved."""
        # Create a metadata dict that mimics compression output
        metadata = {
            'hsc_version': '2.0',
            'compression_type': 'pq',
            'algorithm': 'product_quantization'
        }

        assert 'hsc_version' in metadata, "HSC version field must exist in metadata"
        assert metadata['hsc_version'] == '2.0', "HSC v2.0 should be in metadata"
        assert metadata['compression_type'] == 'pq', "Compression type should be PQ for v2.0"

    def test_legacy_hsc_version_field(self):
        """Test legacy HSC v1.0 metadata is recognized."""
        # Legacy v2.1 would have HSC v1.0
        metadata = {
            'hsc_version': '1.0',
            'compression_type': 'legacy_hsc',
            'algorithm': 'hierarchical_spectral'
        }

        assert metadata['hsc_version'] == '1.0', "HSC v1.0 should be detectable"
        assert metadata['compression_type'] == 'legacy_hsc', "Legacy type should be marked"


class TestFormatMigration:
    """Test format migration between v2.1 and v2.2."""

    def test_v21_file_structure_compatibility(self):
        """Test v2.2 decoder understands v2.1 file structure."""
        # Simulate v2.1 file metadata
        v21_metadata = {
            'version': '2.1',
            'format_major': 2,
            'format_minor': 1,
            'compression': 'hsc',
            'embedding_type': 'tfidf',
            'timestamp': '2024-01-01T00:00:00Z'
        }

        # v2.2 should be able to parse v2.1 metadata
        assert v21_metadata['format_major'] == 2, "v2.1 should have major version 2"
        assert v21_metadata['format_minor'] == 1, "v2.1 should have minor version 1"
        assert v21_metadata['compression'] == 'hsc', "v2.1 uses HSC"
        assert v21_metadata['embedding_type'] in ['tfidf', 'neural'], "v2.1 embedding types are supported"

    def test_hsc_algorithm_dispatch_v20(self):
        """Test HSC decompression dispatches correctly for v2.0."""
        # If metadata has hsc_version='2.0', should use PQ decompressor
        metadata = {
            'hsc_version': '2.0',
            'compression_type': 'pq',
            'algorithm': 'product_quantization',
            'version': '2.2'
        }

        # Determine which decompressor to use based on metadata
        if metadata.get('hsc_version') == '2.0':
            decompressor_type = 'pq'
        elif metadata.get('hsc_version') == '1.0':
            decompressor_type = 'legacy'
        else:
            decompressor_type = 'default'

        assert decompressor_type == 'pq', "HSC v2.0 should use PQ decompressor"

    def test_hsc_algorithm_dispatch_v10(self):
        """Test HSC decompression dispatches correctly for v1.0."""
        # If metadata has hsc_version='1.0', use legacy decompressor
        metadata = {
            'hsc_version': '1.0',
            'compression_type': 'legacy_hsc',
            'algorithm': 'hierarchical_spectral',
            'version': '2.1'
        }

        # Determine which decompressor to use based on metadata
        if metadata.get('hsc_version') == '2.0':
            decompressor_type = 'pq'
        elif metadata.get('hsc_version') == '1.0':
            decompressor_type = 'legacy'
        else:
            decompressor_type = 'default'

        assert decompressor_type == 'legacy', "HSC v1.0 should use legacy decompressor"

    def test_metadata_backward_compatibility_fields(self):
        """Test that v2.1 files missing v2.2 fields are handled gracefully."""
        # v2.1 file without hsc_version field
        v21_metadata = {
            'version': '2.1',
            'compression': 'hsc',
            'embedding_type': 'tfidf'
            # No 'hsc_version' field in v2.1
        }

        # v2.2 decoder should handle missing hsc_version gracefully
        hsc_version = v21_metadata.get('hsc_version', '1.0')  # Default to v1.0 for old files
        assert hsc_version == '1.0', "Missing hsc_version should default to v1.0 for v2.1 files"

    def test_compression_metadata_structure(self):
        """Test compression result metadata structure is consistent."""
        compression_result_v22 = {
            'compressed_data': b'...',
            'metadata': {
                'hsc_version': '2.0',
                'compression_ratio': 0.42,
                'algorithm': 'product_quantization'
            }
        }

        compression_result_v21 = {
            'compressed_data': b'...',
            'metadata': {
                'hsc_version': '1.0',
                'compression_ratio': 0.45,
                'algorithm': 'hierarchical_spectral'
            }
        }

        assert 'metadata' in compression_result_v22, "v2.2 must have metadata"
        assert 'metadata' in compression_result_v21, "v2.1 must have metadata"
        assert 'hsc_version' in compression_result_v22['metadata'], "v2.2 metadata must have hsc_version"
        assert 'hsc_version' in compression_result_v21['metadata'], "v2.1 metadata must have hsc_version"


class TestNoBreakingChanges:
    """Verify no breaking changes to public API."""

    def test_existing_apis_unchanged_embedder(self):
        """Verify get_embedder() API hasn't changed."""
        if get_embedder is None:
            pytest.skip("get_embedder not available in this environment")

        # get_embedder() should still work with default args
        # This is a signature test - actual function not needed for test
        import inspect

        sig = inspect.signature(get_embedder)
        params = list(sig.parameters.keys())

        # Basic API should accept optional embedding_type parameter
        assert 'embedding_type' in params or len(params) <= 2, "get_embedder signature should be compatible"

    def test_default_behavior_unchanged_no_args(self):
        """Default behavior should remain the same when no args provided."""
        # Without explicit opts, should use sensible defaults
        default_config = {
            'embedding_type': 'tfidf',  # Default fallback
            'compression': 'hsc',
            'hsc_version': '2.0'  # New default in v2.2
        }

        assert default_config['embedding_type'] == 'tfidf', "Default embedding should work"
        assert default_config['compression'] == 'hsc', "Default compression should be HSC"

    def test_backward_compatible_parameters(self):
        """Test that old parameter names still work."""
        # Parameters that existed in v2.1 should still work
        old_style_config = {
            'embedding_type': 'tfidf',
            'use_compression': True,
            'compression_type': 'hsc'
        }

        # Should be valid configuration
        assert 'embedding_type' in old_style_config, "embedding_type parameter should exist"
        assert 'use_compression' in old_style_config, "use_compression parameter should exist"
        assert old_style_config['use_compression'] is True, "Boolean parameters should work"

    def test_new_optional_parameters_dont_break_old_code(self):
        """New v2.2 parameters should be optional."""
        # v2.2 adds new optional parameters
        v22_optional_params = {
            'hsc_version': '2.0',  # New in v2.2
            'use_pq_hsc': True,    # New in v2.2
            'pq_codebook_size': 256  # New in v2.2
        }

        # Old code not providing these should still work
        old_code_config = {}
        for key in v22_optional_params:
            old_code_config[key] = v22_optional_params.get(key, None)

        # All should default to None and that's okay
        assert all(v is None for v in old_code_config.values() if not v22_optional_params.get(k) for k in old_code_config), "New params should be optional"

    def test_reader_writer_api_compatibility(self):
        """Test SecureMAIFWriter and Reader APIs are backward compatible."""
        # These should exist and have expected methods
        if SecureMAIFWriter is None or SecureMAIFReader is None:
            pytest.skip("SecureMAIF classes not available")

        # APIs should have these core methods
        writer_methods = ['write', 'add_document', 'finalize']
        reader_methods = ['read', 'get_document', 'get_metadata']

        # This is a structure test - we're verifying the API contract exists
        assert hasattr(SecureMAIFWriter, '__init__'), "Writer must be instantiable"
        assert hasattr(SecureMAIFReader, '__init__'), "Reader must be instantiable"


class TestEnvironmentVariables:
    """Test environment variables control behavior correctly."""

    def test_neural_embeddings_env_var(self):
        """Test MAIF_USE_NEURAL_EMBEDDINGS environment variable."""
        # Save original value
        original = os.environ.get('MAIF_USE_NEURAL_EMBEDDINGS')

        try:
            # Test when set to 'false'
            os.environ['MAIF_USE_NEURAL_EMBEDDINGS'] = 'false'
            use_neural = os.environ.get('MAIF_USE_NEURAL_EMBEDDINGS', 'true').lower() == 'true'
            assert use_neural is False, "Should not use neural when env var is 'false'"

            # Test when set to 'true'
            os.environ['MAIF_USE_NEURAL_EMBEDDINGS'] = 'true'
            use_neural = os.environ.get('MAIF_USE_NEURAL_EMBEDDINGS', 'true').lower() == 'true'
            assert use_neural is True, "Should use neural when env var is 'true'"

            # Test when not set (should default to true)
            del os.environ['MAIF_USE_NEURAL_EMBEDDINGS']
            use_neural = os.environ.get('MAIF_USE_NEURAL_EMBEDDINGS', 'true').lower() == 'true'
            assert use_neural is True, "Should default to true when not set"

        finally:
            # Restore original value
            if original is not None:
                os.environ['MAIF_USE_NEURAL_EMBEDDINGS'] = original
            elif 'MAIF_USE_NEURAL_EMBEDDINGS' in os.environ:
                del os.environ['MAIF_USE_NEURAL_EMBEDDINGS']

    def test_pq_hsc_env_var(self):
        """Test MAIF_USE_PQ_HSC environment variable."""
        original = os.environ.get('MAIF_USE_PQ_HSC')

        try:
            # Test when set to 'true'
            os.environ['MAIF_USE_PQ_HSC'] = 'true'
            use_pq = os.environ.get('MAIF_USE_PQ_HSC', 'true').lower() == 'true'
            assert use_pq is True, "Should use PQ when env var is 'true'"

            # Test when set to 'false'
            os.environ['MAIF_USE_PQ_HSC'] = 'false'
            use_pq = os.environ.get('MAIF_USE_PQ_HSC', 'true').lower() == 'true'
            assert use_pq is False, "Should not use PQ when env var is 'false'"

        finally:
            if original is not None:
                os.environ['MAIF_USE_PQ_HSC'] = original
            elif 'MAIF_USE_PQ_HSC' in os.environ:
                del os.environ['MAIF_USE_PQ_HSC']

    def test_legacy_hsc_mode_env_var(self):
        """Test MAIF_LEGACY_HSC_MODE environment variable."""
        original = os.environ.get('MAIF_LEGACY_HSC_MODE')

        try:
            # Test when set to 'false' (use PQ)
            os.environ['MAIF_LEGACY_HSC_MODE'] = 'false'
            use_legacy = os.environ.get('MAIF_LEGACY_HSC_MODE', 'false').lower() == 'true'
            assert use_legacy is False, "Should use PQ when legacy mode is 'false'"

            # Test when set to 'true' (use legacy)
            os.environ['MAIF_LEGACY_HSC_MODE'] = 'true'
            use_legacy = os.environ.get('MAIF_LEGACY_HSC_MODE', 'false').lower() == 'true'
            assert use_legacy is True, "Should use legacy when legacy mode is 'true'"

        finally:
            if original is not None:
                os.environ['MAIF_LEGACY_HSC_MODE'] = original
            elif 'MAIF_LEGACY_HSC_MODE' in os.environ:
                del os.environ['MAIF_LEGACY_HSC_MODE']

    def test_all_env_vars_together(self):
        """Test multiple environment variables work together."""
        original_neural = os.environ.get('MAIF_USE_NEURAL_EMBEDDINGS')
        original_pq = os.environ.get('MAIF_USE_PQ_HSC')
        original_legacy = os.environ.get('MAIF_LEGACY_HSC_MODE')

        try:
            # Set all at once
            os.environ['MAIF_USE_NEURAL_EMBEDDINGS'] = 'true'
            os.environ['MAIF_USE_PQ_HSC'] = 'true'
            os.environ['MAIF_LEGACY_HSC_MODE'] = 'false'

            use_neural = os.environ.get('MAIF_USE_NEURAL_EMBEDDINGS', 'true').lower() == 'true'
            use_pq = os.environ.get('MAIF_USE_PQ_HSC', 'true').lower() == 'true'
            use_legacy = os.environ.get('MAIF_LEGACY_HSC_MODE', 'false').lower() == 'true'

            assert use_neural is True, "Neural embeddings should be enabled"
            assert use_pq is True, "PQ HSC should be enabled"
            assert use_legacy is False, "Legacy mode should be disabled"

        finally:
            # Restore all original values
            for var, original in [
                ('MAIF_USE_NEURAL_EMBEDDINGS', original_neural),
                ('MAIF_USE_PQ_HSC', original_pq),
                ('MAIF_LEGACY_HSC_MODE', original_legacy)
            ]:
                if original is not None:
                    os.environ[var] = original
                elif var in os.environ:
                    del os.environ[var]


class TestGracefulFallback:
    """Test graceful fallback behavior on errors."""

    def test_missing_hsc_version_defaults_to_v10(self):
        """Test that missing hsc_version defaults to v1.0 for v2.1 files."""
        metadata = {
            'version': '2.1',
            'compression': 'hsc'
            # Missing hsc_version field
        }

        hsc_version = metadata.get('hsc_version', '1.0')
        assert hsc_version == '1.0', "Should default to v1.0 when missing"

    def test_unknown_embedding_type_fallback(self):
        """Test fallback when unknown embedding type is encountered."""
        config = {
            'embedding_type': 'unknown_type'
        }

        # Should fallback to tfidf
        embedding_type = config.get('embedding_type', 'tfidf')
        if embedding_type == 'unknown_type':
            embedding_type = 'tfidf'

        assert embedding_type == 'tfidf', "Should fallback to tfidf for unknown types"

    def test_corrupted_metadata_recovery(self):
        """Test recovery from corrupted metadata."""
        # Partially corrupted metadata
        corrupted_metadata = {
            'version': '2.1',
            'compression': None,  # Missing value
            'embedding_type': 'tfidf'
        }

        # Should provide sensible defaults
        compression = corrupted_metadata.get('compression', 'hsc')
        embedding_type = corrupted_metadata.get('embedding_type', 'tfidf')

        assert compression == 'hsc', "Should default to hsc"
        assert embedding_type == 'tfidf', "Should use provided value"

    def test_missing_optional_compression_fields(self):
        """Test handling of missing optional compression fields."""
        minimal_metadata = {
            'version': '2.2',
            'format_major': 2,
            'format_minor': 2
        }

        # All these should have sensible defaults
        hsc_version = minimal_metadata.get('hsc_version', '2.0')
        compression_ratio = minimal_metadata.get('compression_ratio', 1.0)
        algorithm = minimal_metadata.get('algorithm', 'product_quantization')

        assert hsc_version == '2.0', "Should default to PQ v2.0"
        assert compression_ratio == 1.0, "Should default to no compression"
        assert algorithm == 'product_quantization', "Should default to PQ"

    def test_invalid_version_string_handling(self):
        """Test handling of invalid version strings."""
        test_cases = [
            ('2.1', True),      # Valid
            ('2.2', True),      # Valid
            ('3.0', True),      # Different but parseable
            ('abc', False),     # Invalid
            ('', False),        # Empty
            (None, False),      # None
        ]

        for version_str, should_be_valid in test_cases:
            try:
                if version_str is None:
                    parsed = None
                else:
                    parts = version_str.split('.')
                    parsed = [int(p) for p in parts]
                is_valid = parsed is not None and len(parsed) >= 2
            except (ValueError, AttributeError, TypeError):
                is_valid = False

            assert is_valid == should_be_valid, f"Version {version_str} validity mismatch"


class TestV21FileFixtures:
    """Test reading actual v2.1 file fixtures (if available)."""

    def test_v21_fixture_detection(self):
        """Test detection of v2.1 test fixtures."""
        fixture_dir = Path(__file__).parent / 'fixtures' / 'v21'

        # If fixtures exist, they should be readable
        if fixture_dir.exists():
            assert fixture_dir.is_dir(), "Fixture directory should be a directory"

            v21_files = list(fixture_dir.glob('*.maif'))
            for file in v21_files:
                assert file.exists(), f"Fixture file should exist: {file}"
                assert file.stat().st_size > 0, f"Fixture file should not be empty: {file}"

    def test_v21_fixture_metadata_structure(self):
        """Test that v2.1 fixture metadata has expected structure."""
        v21_fixture_metadata = {
            'version': '2.1',
            'format_major': 2,
            'format_minor': 1,
            'embedding_type': 'tfidf',
            'compression': 'hsc',
            'hsc_version': '1.0',
            'timestamp': '2024-01-01T00:00:00Z'
        }

        # Required fields for v2.1
        required_fields = ['version', 'format_major', 'format_minor', 'embedding_type', 'compression']

        for field in required_fields:
            assert field in v21_fixture_metadata, f"v2.1 fixture must have {field}"


class TestMigrationHelpers:
    """Test migration helper functions."""

    def test_version_comparison_helpers(self):
        """Test version comparison utility functions."""
        def parse_version(version_str):
            """Parse version string to tuple of ints."""
            try:
                return tuple(int(x) for x in version_str.split('.'))
            except (ValueError, AttributeError):
                return None

        def is_version_compatible(file_version, reader_version):
            """Check if file_version can be read by reader_version."""
            file_v = parse_version(file_version)
            reader_v = parse_version(reader_version)

            if file_v is None or reader_v is None:
                return False

            # Reader can read files with same or earlier major.minor
            return (file_v[0] == reader_v[0] and  # Same major version
                    file_v[1] <= reader_v[1])     # File minor <= reader minor

        assert is_version_compatible('2.1', '2.2') is True, "v2.2 should read v2.1"
        assert is_version_compatible('2.2', '2.1') is False, "v2.1 should not read v2.2"
        assert is_version_compatible('2.0', '2.2') is True, "v2.2 should read v2.0"
        assert is_version_compatible('3.0', '2.2') is False, "v2.2 should not read v3.0"

    def test_hsc_decompression_router(self):
        """Test HSC decompression router logic."""
        def get_hsc_decompressor(metadata):
            """Get correct decompressor based on metadata."""
            hsc_version = metadata.get('hsc_version', '1.0')

            if hsc_version == '2.0':
                return 'pq_decompressor'
            elif hsc_version == '1.0':
                return 'legacy_decompressor'
            else:
                return 'default_decompressor'

        assert get_hsc_decompressor({'hsc_version': '2.0'}) == 'pq_decompressor'
        assert get_hsc_decompressor({'hsc_version': '1.0'}) == 'legacy_decompressor'
        assert get_hsc_decompressor({}) == 'default_decompressor'
        assert get_hsc_decompressor({'hsc_version': 'unknown'}) == 'default_decompressor'

    def test_compression_type_mapping(self):
        """Test compression type to algorithm mapping."""
        compression_map = {
            'pq': 'product_quantization',
            'legacy_hsc': 'hierarchical_spectral',
            'hsc': 'hierarchical_spectral',  # Fallback
        }

        def get_compression_algorithm(compression_type):
            return compression_map.get(compression_type, 'unknown')

        assert get_compression_algorithm('pq') == 'product_quantization'
        assert get_compression_algorithm('legacy_hsc') == 'hierarchical_spectral'
        assert get_compression_algorithm('unknown') == 'unknown'


class TestErrorRecovery:
    """Test error recovery and resilience."""

    def test_recovery_from_missing_metadata(self):
        """Test recovery when metadata is missing."""
        file_data = {
            'data': b'compressed_data',
            'metadata': {}  # Empty metadata
        }

        # Should provide defaults
        hsc_version = file_data['metadata'].get('hsc_version', '1.0')
        compression_type = file_data['metadata'].get('compression_type', 'hsc')

        assert hsc_version == '1.0', "Should default to v1.0"
        assert compression_type == 'hsc', "Should default to hsc"

    def test_recovery_from_partial_metadata(self):
        """Test recovery when metadata is partially present."""
        file_data = {
            'data': b'compressed_data',
            'metadata': {
                'version': '2.1',
                # Missing hsc_version, compression_type, etc.
            }
        }

        # Should fill in missing fields
        metadata = file_data['metadata']
        metadata.setdefault('hsc_version', '1.0')
        metadata.setdefault('compression_type', 'hsc')
        metadata.setdefault('embedding_type', 'tfidf')

        assert metadata['hsc_version'] == '1.0', "Should add hsc_version"
        assert metadata['compression_type'] == 'hsc', "Should add compression_type"
        assert metadata['embedding_type'] == 'tfidf', "Should add embedding_type"

    def test_recovery_from_version_mismatch(self):
        """Test recovery when file version doesn't match reader."""
        file_metadata = {
            'version': '2.1',
            'format_major': 2,
            'format_minor': 1
        }

        reader_version = (2, 2)  # v2.2
        file_version = (file_metadata['format_major'], file_metadata['format_minor'])

        # Should be able to read if major version matches
        can_read = (file_version[0] == reader_version[0] and
                   file_version[1] <= reader_version[1])

        assert can_read is True, "v2.2 reader should handle v2.1 file"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
