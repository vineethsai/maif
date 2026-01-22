"""
MAIF Document Loader for LangChain.

Provides a LangChain-compatible document loader that reads MAIF files
and extracts text blocks as Documents with preserved metadata, signatures,
and provenance information.
"""

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Union
from pathlib import Path

try:
    from langchain_core.document_loaders.base import BaseLoader
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseLoader = object
    Document = dict


class MAIFLoader(BaseLoader if LANGCHAIN_AVAILABLE else object):
    """MAIF Document Loader for LangChain.

    Loads MAIF files and extracts text blocks as LangChain Documents.
    Preserves all metadata including cryptographic signatures, provenance
    chain, and block-level metadata for full audit trail support.

    Usage:
        from maif.integrations.langchain import MAIFLoader

        # Load a single MAIF file
        loader = MAIFLoader("document.maif")
        documents = loader.load()

        # Lazy loading for large files
        for doc in loader.lazy_load():
            process(doc)

        # Load multiple files
        loader = MAIFLoader(["file1.maif", "file2.maif"])
        documents = loader.load()

        # Access provenance metadata
        for doc in documents:
            print(f"Content: {doc.page_content[:100]}...")
            print(f"Block ID: {doc.metadata.get('block_id')}")
            print(f"Signature: {doc.metadata.get('signature')}")
            print(f"Provenance: {doc.metadata.get('provenance')}")

    Metadata Fields:
        - source: Path to the source MAIF file
        - block_id: Unique identifier for the block within the file
        - block_index: Zero-based index of the block
        - block_type: Type of the block (text, binary, etc.)
        - signature: Ed25519 signature of the block (if present)
        - signature_valid: Whether the signature verification passed
        - provenance: Provenance chain information
        - agent_id: ID of the agent that created the block
        - timestamp: When the block was created
        - content_hash: SHA-256 hash of the content
        - custom_metadata: Any custom metadata from the block
    """

    def __init__(
        self,
        file_path: Union[str, Path, List[Union[str, Path]]],
        extract_binary: bool = False,
        verify_signatures: bool = True,
        include_provenance: bool = True,
        text_blocks_only: bool = True,
        encoding: str = "utf-8",
    ):
        """Initialize the MAIF loader.

        Args:
            file_path: Path to a single MAIF file, or list of paths
            extract_binary: If True, attempt to decode binary blocks as text
            verify_signatures: If True, verify block signatures and include result
            include_provenance: If True, include full provenance chain in metadata
            text_blocks_only: If True, only load text-type blocks (skip binary)
            encoding: Text encoding to use when decoding binary blocks
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for MAIFLoader. "
                "Install with: pip install langchain-core"
            )

        # Normalize to list of paths
        if isinstance(file_path, (str, Path)):
            self.file_paths = [Path(file_path)]
        else:
            self.file_paths = [Path(p) for p in file_path]

        self.extract_binary = extract_binary
        self.verify_signatures = verify_signatures
        self.include_provenance = include_provenance
        self.text_blocks_only = text_blocks_only
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load all documents from the MAIF file(s).

        Returns:
            List of Document objects with content and metadata
        """
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the MAIF file(s).

        Yields:
            Document objects one at a time for memory efficiency
        """
        for file_path in self.file_paths:
            yield from self._load_single_file(file_path)

    def _load_single_file(self, file_path: Path) -> Iterator[Document]:
        """Load documents from a single MAIF file.

        Args:
            file_path: Path to the MAIF file

        Yields:
            Document objects from the file
        """
        from maif import MAIFDecoder

        if not file_path.exists():
            raise FileNotFoundError(f"MAIF file not found: {file_path}")

        decoder = MAIFDecoder(str(file_path))
        decoder.load()

        # Optionally verify integrity
        file_valid = True
        verification_errors = []
        if self.verify_signatures:
            file_valid, verification_errors = decoder.verify_integrity()

        # Get provenance chain if requested
        provenance_chain = []
        if self.include_provenance:
            provenance_chain = self._extract_provenance(decoder)

        # Process each block
        for block_index, block in enumerate(decoder.blocks):
            doc = self._block_to_document(
                block=block,
                block_index=block_index,
                file_path=file_path,
                file_valid=file_valid,
                provenance_chain=provenance_chain,
            )
            if doc is not None:
                yield doc

    def _block_to_document(
        self,
        block: Any,
        block_index: int,
        file_path: Path,
        file_valid: bool,
        provenance_chain: List[Dict[str, Any]],
    ) -> Optional[Document]:
        """Convert a MAIF block to a LangChain Document.

        Args:
            block: The MAIF block object
            block_index: Index of the block in the file
            file_path: Source file path
            file_valid: Whether file integrity verification passed
            provenance_chain: Extracted provenance information

        Returns:
            Document object or None if block should be skipped
        """
        # Get block type
        block_type = getattr(block, "block_type", None)
        if block_type is not None:
            block_type = str(block_type.name if hasattr(block_type, "name") else block_type)

        # Skip binary blocks if configured
        is_text_block = block_type in ("TEXT", "text", "TEXT_BLOCK", None)
        if self.text_blocks_only and not is_text_block and not self.extract_binary:
            return None

        # Extract content
        content = self._extract_content(block)
        if content is None:
            return None

        # Build metadata
        metadata = self._build_metadata(
            block=block,
            block_index=block_index,
            block_type=block_type,
            file_path=file_path,
            file_valid=file_valid,
            provenance_chain=provenance_chain,
            content=content,
        )

        return Document(page_content=content, metadata=metadata)

    def _extract_content(self, block: Any) -> Optional[str]:
        """Extract text content from a block.

        Args:
            block: The MAIF block object

        Returns:
            Text content or None if extraction fails
        """
        data = getattr(block, "data", None)
        if data is None:
            return None

        # Handle bytes
        if isinstance(data, bytes):
            if self.extract_binary or self._looks_like_text(data):
                try:
                    return data.decode(self.encoding)
                except UnicodeDecodeError:
                    if self.extract_binary:
                        # Return as hex if we're extracting binary anyway
                        return data.hex()
                    return None
            return None

        # Handle string
        if isinstance(data, str):
            return data

        # Handle dict/other - serialize to JSON
        try:
            return json.dumps(data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(data)

    def _looks_like_text(self, data: bytes) -> bool:
        """Check if bytes look like text content.

        Args:
            data: Bytes to check

        Returns:
            True if content appears to be text
        """
        if not data:
            return False

        # Sample the first 1000 bytes
        sample = data[:1000]

        # Check for null bytes (common in binary)
        if b"\x00" in sample:
            return False

        # Try to decode as UTF-8
        try:
            sample.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    def _build_metadata(
        self,
        block: Any,
        block_index: int,
        block_type: Optional[str],
        file_path: Path,
        file_valid: bool,
        provenance_chain: List[Dict[str, Any]],
        content: str,
    ) -> Dict[str, Any]:
        """Build metadata dictionary for a document.

        Args:
            block: The MAIF block object
            block_index: Index of the block
            block_type: Type of the block
            file_path: Source file path
            file_valid: Whether file verification passed
            provenance_chain: Provenance information
            content: Extracted content (for hashing)

        Returns:
            Metadata dictionary
        """
        import hashlib

        metadata: Dict[str, Any] = {
            "source": str(file_path.absolute()),
            "block_index": block_index,
            "block_type": block_type,
        }

        # Block ID
        block_id = getattr(block, "id", None) or getattr(block, "block_id", None)
        if block_id:
            metadata["block_id"] = str(block_id)
        else:
            metadata["block_id"] = f"{file_path.stem}_{block_index}"

        # Signature information
        signature = getattr(block, "signature", None)
        if signature:
            if isinstance(signature, bytes):
                metadata["signature"] = signature.hex()
            else:
                metadata["signature"] = str(signature)

        # Signature validity
        if self.verify_signatures:
            metadata["signature_valid"] = file_valid

        # Content hash
        metadata["content_hash"] = hashlib.sha256(content.encode()).hexdigest()

        # Timestamp
        timestamp = getattr(block, "timestamp", None)
        if timestamp:
            metadata["timestamp"] = timestamp

        # Agent ID
        agent_id = getattr(block, "agent_id", None)
        if agent_id:
            metadata["agent_id"] = agent_id

        # Block-level metadata
        block_metadata = getattr(block, "metadata", None)
        if block_metadata:
            if isinstance(block_metadata, dict):
                metadata["custom_metadata"] = block_metadata
            else:
                try:
                    metadata["custom_metadata"] = dict(block_metadata)
                except (TypeError, ValueError):
                    metadata["custom_metadata"] = str(block_metadata)

        # Provenance chain
        if self.include_provenance and provenance_chain:
            # Find provenance entry for this block
            block_provenance = None
            for entry in provenance_chain:
                if entry.get("block_index") == block_index:
                    block_provenance = entry
                    break

            if block_provenance:
                metadata["provenance"] = block_provenance
            else:
                # Include full chain reference
                metadata["provenance_chain_length"] = len(provenance_chain)

        return metadata

    def _extract_provenance(self, decoder: Any) -> List[Dict[str, Any]]:
        """Extract provenance chain from the decoder.

        Args:
            decoder: The MAIFDecoder instance

        Returns:
            List of provenance entries as dictionaries
        """
        provenance_chain = []

        # Try to get provenance from various possible attributes
        provenance = (
            getattr(decoder, "provenance_chain", None) or
            getattr(decoder, "provenance", None) or
            getattr(decoder, "provenance_entries", None)
        )

        if provenance:
            for i, entry in enumerate(provenance):
                entry_dict = {
                    "block_index": i,
                }

                # Extract common fields
                for field in ["hash", "prev_hash", "timestamp", "agent_id", "action", "signature"]:
                    value = getattr(entry, field, None)
                    if value is not None:
                        if isinstance(value, bytes):
                            entry_dict[field] = value.hex()
                        else:
                            entry_dict[field] = value

                provenance_chain.append(entry_dict)

        return provenance_chain

    def get_file_info(self) -> List[Dict[str, Any]]:
        """Get information about the MAIF files without loading content.

        Returns:
            List of file information dictionaries
        """
        from maif import MAIFDecoder

        info_list = []

        for file_path in self.file_paths:
            if not file_path.exists():
                info_list.append({
                    "path": str(file_path),
                    "exists": False,
                    "error": "File not found",
                })
                continue

            try:
                decoder = MAIFDecoder(str(file_path))
                decoder.load()

                # Verify if requested
                valid = True
                errors = []
                if self.verify_signatures:
                    valid, errors = decoder.verify_integrity()

                info_list.append({
                    "path": str(file_path.absolute()),
                    "exists": True,
                    "block_count": len(decoder.blocks),
                    "valid": valid,
                    "verification_errors": errors if errors else None,
                    "file_size": file_path.stat().st_size,
                })
            except Exception as e:
                info_list.append({
                    "path": str(file_path),
                    "exists": True,
                    "error": str(e),
                })

        return info_list
