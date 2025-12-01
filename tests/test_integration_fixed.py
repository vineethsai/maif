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

from maif.integration_enhanced import ConversionResult, EnhancedMAIFProcessor
from maif.integration import MAIFConverter, MAIFPluginManager
from maif.core import MAIFEncoder, MAIFDecoder, MAIFParser


class TestConversionResult:
    """Test ConversionResult data structure."""
    
    def test_conversion_result_creation(self):
        """Test basic ConversionResult creation."""
        result = ConversionResult(
            success=True,
            input_path="/path/to/input.json",
            output_path="/path/to/output.maif",
            warnings=["Warning 1", "Warning 2"],
            metadata={"blocks_converted": 5, "format": "json"}
        )
        
        assert result.success is True
        assert result.output_path == "/path/to/output.maif"
        assert result.warnings == ["Warning 1", "Warning 2"]
        assert result.metadata["blocks_converted"] == 5
        assert result.metadata["format"] == "json"


class TestMAIFConverter:
    """Test MAIFConverter functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.converter = MAIFConverter()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_converter_initialization(self):
        """Test MAIFConverter initialization."""
        assert hasattr(self.converter, 'supported_formats')
        assert 'json' in self.converter.supported_formats
        assert 'xml' in self.converter.supported_formats
        assert 'csv' in self.converter.supported_formats
        assert 'txt' in self.converter.supported_formats
    
    def test_convert_json_to_maif(self):
        """Test JSON to MAIF conversion."""
        # Create test JSON file
        test_data = {
            "title": "Test Document",
            "content": "This is test content for JSON conversion",
            "metadata": {
                "author": "Test Author",
                "created": "2024-01-01"
            }
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
    
    def test_convert_xml_to_maif(self):
        """Test XML to MAIF conversion."""
        # Create test XML file
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<document>
    <title>Test XML Document</title>
    <content>This is test content for XML conversion</content>
    <metadata>
        <author>Test Author</author>
        <created>2024-01-01</created>
    </metadata>
</document>'''
        
        xml_path = os.path.join(self.temp_dir, "test.xml")
        with open(xml_path, 'w') as f:
            f.write(xml_content)
        
        maif_path = os.path.join(self.temp_dir, "converted_xml.maif")
        manifest_path = os.path.join(self.temp_dir, "converted_xml_manifest.json")
        
        result = self.converter.convert_to_maif(
            input_path=xml_path,
            output_path=maif_path,
            manifest_path=manifest_path,
            input_format="xml"
        )
        
        assert result.success is True
        assert os.path.exists(maif_path)
        assert os.path.exists(manifest_path)
    
    def test_export_maif_to_json(self):
        """Test MAIF to JSON export."""
        # Create a test MAIF file
        encoder = MAIFEncoder(agent_id="test_agent")
        encoder.add_text_block("Test content for export", metadata={"id": 1})
        
        maif_path = os.path.join(self.temp_dir, "export_test.maif")
        manifest_path = os.path.join(self.temp_dir, "export_test_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Export to JSON
        json_output = os.path.join(self.temp_dir, "exported.json")
        
        result = self.converter.export_from_maif(
            maif_path=maif_path,
            output_path=json_output,
            manifest_path=manifest_path,
            output_format="json"
        )
        
        assert result.success is True
        assert os.path.exists(json_output)


class TestMAIFPluginManager:
    """Test MAIFPluginManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plugin_manager = MAIFPluginManager()
    
    def test_plugin_manager_initialization(self):
        """Test MAIFPluginManager initialization."""
        assert hasattr(self.plugin_manager, 'plugins')
        assert hasattr(self.plugin_manager, 'hooks')
        assert self.plugin_manager.plugins == []
        # Check that hooks are initialized with expected hook names
        expected_hooks = {
            "pre_conversion": [],
            "post_conversion": [],
            "pre_validation": [],
            "post_validation": []
        }
        assert self.plugin_manager.hooks == expected_hooks
    
    def test_register_hook(self):
        """Test hook registration."""
        def test_callback(data):
            return f"processed: {data}"
        
        self.plugin_manager.register_hook("test_hook", test_callback)
        
        assert "test_hook" in self.plugin_manager.hooks
        assert test_callback in self.plugin_manager.hooks["test_hook"]
    
    def test_execute_hooks(self):
        """Test hook execution."""
        results = []
        
        def callback1(data):
            results.append(f"callback1: {data}")
            return f"result1: {data}"
        
        def callback2(data):
            results.append(f"callback2: {data}")
            return f"result2: {data}"
        
        # Register hooks
        self.plugin_manager.register_hook("test_hook", callback1)
        self.plugin_manager.register_hook("test_hook", callback2)
        
        # Execute hooks
        hook_results = self.plugin_manager.execute_hooks("test_hook", "test_data")
        
        assert len(results) == 2
        assert "callback1: test_data" in results
        assert "callback2: test_data" in results
        assert len(hook_results) == 2


if __name__ == "__main__":
    pytest.main([__file__])