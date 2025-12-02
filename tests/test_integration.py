"""
Comprehensive tests for MAIF integration functionality.
"""

import pytest
import tempfile
import os
import json
import zipfile
import tarfile
from unittest.mock import Mock, patch, MagicMock

from maif.integration import ConversionResult, EnhancedMAIFProcessor
from maif.core import MAIFEncoder, MAIFDecoder, MAIFParser


class TestConversionResult:
    """Test ConversionResult data structure."""
    
    def test_conversion_result_creation(self):
        """Test basic ConversionResult creation."""
        result = ConversionResult(
            success=True,
            output_path="/path/to/output.maif",
            warnings=["Warning 1", "Warning 2"],
            metadata={"blocks_converted": 5, "format": "json"}
        )
        
        assert result.success is True
        assert result.output_path == "/path/to/output.maif"
        assert result.warnings == ["Warning 1", "Warning 2"]
        assert result.metadata["blocks_converted"] == 5
        assert result.metadata["format"] == "json"
    
    def test_conversion_result_post_init(self):
        """Test ConversionResult post-initialization processing."""
        result = ConversionResult(
            success=True,
            output_path="/path/to/output.maif"
        )
        
        # Should initialize empty collections
        assert result.warnings == []
        assert result.metadata == {}


class TestEnhancedMAIFProcessor:
    """Test EnhancedMAIFProcessor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.integration = EnhancedMAIFProcessor()
        self.converter = self.integration  # Alias for test compatibility
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_processor_initialization(self):
        """Test EnhancedMAIFProcessor initialization."""
        assert hasattr(self.integration, 'supported_formats')
        assert 'json' in self.integration.supported_formats
        assert 'xml' in self.integration.supported_formats
        assert 'csv' in self.integration.supported_formats
        assert 'txt' in self.integration.supported_formats
    
    def test_mime_to_format_conversion(self):
        """Test MIME type to format conversion."""
        test_cases = [
            ("application/json", "json"),
            ("text/xml", "xml"),
            ("application/xml", "xml"),
            ("text/csv", "csv"),
            ("text/plain", "txt"),
            ("application/zip", "zip"),
            ("application/x-tar", "tar")
        ]
        
        for mime_type, expected_format in test_cases:
            result = self.integration._mime_to_format(mime_type)
            assert result == expected_format
    
    def test_convert_json_to_maif(self):
        """Test JSON to MAIF conversion."""
        # Create test JSON file
        test_data = {
            "title": "Test Document",
            "content": "This is test content for JSON conversion",
            "metadata": {
                "author": "Test Author",
                "created": "2024-01-01"
            },
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"}
            ]
        }
        
        json_path = os.path.join(self.temp_dir, "test.json")
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        maif_path = os.path.join(self.temp_dir, "converted.maif")
        manifest_path = os.path.join(self.temp_dir, "converted_manifest.json")
        
        result = self.converter.convert_to_maif(
            input_path=json_path,
            output_path=maif_path,
            manifest_path=manifest_path,
            input_format="json"
        )
        
        assert result.success is True
        assert os.path.exists(maif_path)
        assert os.path.exists(manifest_path)
        
        # Verify conversion by reading back
        parser = MAIFParser(maif_path, manifest_path)
        content = parser.extract_content()
        
        assert "text_blocks" in content
        assert len(content["text_blocks"]) > 0
    
    def test_convert_xml_to_maif(self):
        """Test XML to MAIF conversion."""
        # Create test XML file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <document>
            <title>Test XML Document</title>
            <content>This is test content for XML conversion</content>
        </document>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as xml_file:
            xml_file.write(xml_content)
            xml_file_path = xml_file.name
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "converted.maif")
                manifest_path = os.path.join(temp_dir, "converted_manifest.json")
                
                # Convert XML to MAIF
                result = self.integration.convert_xml_to_maif(xml_file_path, output_path, manifest_path)
                
                assert result.success is True
                assert os.path.exists(output_path)
                assert os.path.exists(manifest_path)
                
        finally:
            os.unlink(xml_file_path)