/**
 * MAIF File Format Parser
 * Supports both legacy format (with manifest) and new secure format
 */

class MAIFParser {
    constructor() {
        this.manifest = null;
        this.binaryData = null;
        this.blocks = [];
        this.provenance = [];
        this.securityInfo = null;
        this.fileInfo = null;
        this.format = 'unknown'; // 'legacy' or 'secure'
    }

    // Block type mappings
    static BLOCK_TYPES = {
        'TEXT': { name: 'Text', icon: 'T', color: '#3b82f6', code: 0x54455854 },
        'EMBD': { name: 'Embeddings', icon: 'E', color: '#8b5cf6', code: 0x454D4244 },
        'KGRF': { name: 'Knowledge Graph', icon: 'K', color: '#ec4899', code: 0x4B4E4F57 },
        'KNOW': { name: 'Knowledge Graph', icon: 'K', color: '#ec4899', code: 0x4B4E4F57 },
        'IMAG': { name: 'Image', icon: 'I', color: '#10b981', code: 0x494D4147 },
        'AUDI': { name: 'Audio', icon: 'A', color: '#f59e0b', code: 0x41554449 },
        'VIDE': { name: 'Video', icon: 'V', color: '#ef4444', code: 0x56494445 },
        'SECU': { name: 'Security', icon: 'S', color: '#6366f1', code: 0x53454355 },
        'LIFE': { name: 'Lifecycle', icon: 'L', color: '#14b8a6', code: 0x4C494645 },
        'COMP': { name: 'Compressed', icon: 'C', color: '#64748b', code: 0x434F4D50 },
        'ENCR': { name: 'Encrypted', icon: 'X', color: '#f97316', code: 0x454E4352 },
        'BINA': { name: 'Binary', icon: 'B', color: '#64748b', code: 0x42494E41 },
        'META': { name: 'Metadata', icon: 'M', color: '#94a3b8', code: 0x4D455441 }
    };

    // Secure format constants
    static SECURE_HEADER_SIZE = 444;
    static SECURE_BLOCK_HEADER_SIZE = 372;
    static FOOTER_SIZE = 48;
    static MAGIC_HEADER = 'MAIF';
    static MAGIC_FOOTER = 'FIAM';

    /**
     * Load and parse a manifest JSON file (legacy format)
     */
    async loadManifest(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    this.manifest = JSON.parse(e.target.result);
                    this.format = 'legacy';
                    this.parseManifest();
                    resolve(this.manifest);
                } catch (err) {
                    reject(new Error(`Failed to parse manifest: ${err.message}`));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read manifest file'));
            reader.readAsText(file);
        });
    }

    /**
     * Load and parse a .maif binary file
     */
    async loadBinary(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.binaryData = new Uint8Array(e.target.result);
                this.detectAndParseBinary();
                resolve(this.binaryData);
            };
            reader.onerror = () => reject(new Error('Failed to read binary file'));
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Detect format and parse binary accordingly
     */
    detectAndParseBinary() {
        if (!this.binaryData || this.binaryData.length < 4) return;

        // Check magic number
        const magic = String.fromCharCode(...this.binaryData.slice(0, 4));
        
        if (magic === MAIFParser.MAGIC_HEADER) {
            // New secure format
            this.format = 'secure';
            this.parseSecureBinary();
        } else {
            // Legacy format (raw blocks)
            if (this.format !== 'legacy') {
                this.format = 'legacy';
            }
            this.parseLegacyBinary();
        }
    }

    /**
     * Parse secure format binary
     */
    parseSecureBinary() {
        if (!this.binaryData) return;

        try {
            // Parse file header (444 bytes)
            const headerView = new DataView(this.binaryData.buffer);
            
            const magic = String.fromCharCode(...this.binaryData.slice(0, 4));
            if (magic !== MAIFParser.MAGIC_HEADER) {
                throw new Error('Invalid magic number');
            }

            const versionMajor = headerView.getUint16(4, false);
            const versionMinor = headerView.getUint16(6, false);
            const flags = headerView.getUint32(8, false);
            const created = this.readUint64(headerView, 12);
            const modified = this.readUint64(headerView, 20);
            const fileId = this.bytesToHex(this.binaryData.slice(28, 44));
            const creatorId = this.bytesToHex(this.binaryData.slice(44, 60));
            const agentDid = this.nullTerminatedString(this.binaryData.slice(60, 124));
            const blockCount = headerView.getUint32(124, false);
            const provenanceOffset = this.readUint64(headerView, 128);
            const securityOffset = this.readUint64(headerView, 136);
            const indexOffset = this.readUint64(headerView, 144);
            const merkleRoot = this.bytesToHex(this.binaryData.slice(152, 184));
            const fileSignature = this.bytesToHex(this.binaryData.slice(184, 440));

            this.fileInfo = {
                format: 'secure',
                version: `${versionMajor}.${versionMinor}`,
                flags,
                created,
                modified,
                fileId,
                creatorId,
                agentDid,
                blockCount,
                provenanceOffset,
                securityOffset,
                indexOffset,
                merkleRoot,
                fileSignature,
                isSigned: !!(flags & 0x01),
                isFinalized: !!(flags & 0x02)
            };

            // Parse blocks
            this.blocks = [];
            let offset = MAIFParser.SECURE_HEADER_SIZE;

            for (let i = 0; i < blockCount; i++) {
                const block = this.parseSecureBlock(offset, i);
                if (block) {
                    this.blocks.push(block);
                    offset += block.size;
                }
            }

            // Parse provenance section
            this.parseProvenanceSection(provenanceOffset);

            // Parse security section
            this.parseSecuritySection(securityOffset);

            // Build manifest-like structure for compatibility
            this.buildCompatibilityManifest();

        } catch (err) {
            console.error('Error parsing secure format:', err);
        }
    }

    /**
     * Parse a single secure format block
     */
    parseSecureBlock(offset, index) {
        if (offset + MAIFParser.SECURE_BLOCK_HEADER_SIZE > this.binaryData.length) {
            return null;
        }

        const view = new DataView(this.binaryData.buffer, offset);
        
        const size = view.getUint32(0, false);
        const blockTypeCode = view.getUint32(4, false);
        const flags = view.getUint32(8, false);
        const version = view.getUint32(12, false);
        const timestamp = this.readUint64(view, 16);
        const blockId = this.bytesToHex(this.binaryData.slice(offset + 24, offset + 40));
        const previousHash = this.bytesToHex(this.binaryData.slice(offset + 40, offset + 72));
        const contentHash = this.bytesToHex(this.binaryData.slice(offset + 72, offset + 104));
        const signature = this.bytesToHex(this.binaryData.slice(offset + 104, offset + 360));
        const metadataSize = view.getUint32(360, false);

        // Convert block type code to string
        const type = this.blockTypeCodeToString(blockTypeCode);
        
        // Calculate data offset and size
        const dataOffset = offset + MAIFParser.SECURE_BLOCK_HEADER_SIZE;
        const dataSize = size - MAIFParser.SECURE_BLOCK_HEADER_SIZE - metadataSize;

        // Read metadata if present
        let metadata = null;
        if (metadataSize > 0) {
            try {
                const metaBytes = this.binaryData.slice(dataOffset + dataSize, dataOffset + dataSize + metadataSize);
                const metaStr = new TextDecoder().decode(metaBytes);
                metadata = JSON.parse(metaStr);
            } catch (e) {
                metadata = { _error: 'Failed to parse metadata' };
            }
        }

        return {
            index,
            type,
            block_type: type,
            offset,
            size,
            dataOffset,
            dataSize,
            hash: contentHash,
            content_hash: contentHash,
            version,
            timestamp,
            block_id: blockId,
            previous_hash: previousHash,
            signature,
            flags,
            metadata_size: metadataSize,
            metadata,
            isSigned: !!(flags & 0x01),
            isImmutable: !!(flags & 0x02),
            isTampered: !!(flags & 0x20),
            typeInfo: MAIFParser.BLOCK_TYPES[type] || { name: type, icon: '?', color: '#64748b' }
        };
    }

    /**
     * Parse provenance section
     */
    parseProvenanceSection(offset) {
        if (offset === 0 || offset >= this.binaryData.length) {
            this.provenance = [];
            return;
        }

        try {
            const view = new DataView(this.binaryData.buffer, offset);
            const size = view.getUint32(0, false);
            const data = this.binaryData.slice(offset + 4, offset + 4 + size);
            const jsonStr = new TextDecoder().decode(data);
            this.provenance = JSON.parse(jsonStr);
        } catch (e) {
            console.error('Error parsing provenance:', e);
            this.provenance = [];
        }
    }

    /**
     * Parse security section
     */
    parseSecuritySection(offset) {
        if (offset === 0 || offset >= this.binaryData.length) {
            this.securityInfo = null;
            return;
        }

        try {
            const view = new DataView(this.binaryData.buffer, offset);
            const size = view.getUint32(0, false);
            const data = this.binaryData.slice(offset + 4, offset + 4 + size);
            const jsonStr = new TextDecoder().decode(data);
            this.securityInfo = JSON.parse(jsonStr);
        } catch (e) {
            console.error('Error parsing security section:', e);
            this.securityInfo = null;
        }
    }

    /**
     * Build a manifest-like structure for UI compatibility
     */
    buildCompatibilityManifest() {
        this.manifest = {
            maif_version: this.fileInfo?.version || '2.0',
            format: 'secure',
            created: this.fileInfo?.created,
            agent_id: this.fileInfo?.agentDid,
            file_id: this.fileInfo?.fileId,
            blocks: this.blocks.map(b => ({
                type: b.type,
                block_type: b.type,
                offset: b.offset,
                size: b.size,
                hash: b.hash,
                version: b.version,
                block_id: b.block_id,
                previous_hash: b.previous_hash,
                metadata: b.metadata,
                is_signed: b.isSigned,
                is_tampered: b.isTampered
            })),
            signature: this.fileInfo?.fileSignature,
            public_key: this.securityInfo?.public_key,
            root_hash: this.fileInfo?.merkleRoot,
            signature_metadata: {
                signer_id: this.securityInfo?.signer_id,
                signer_did: this.securityInfo?.signer_did,
                timestamp: this.securityInfo?.signed_at,
                provenance_chain: this.provenance
            }
        };
    }

    /**
     * Parse legacy binary format (raw blocks without file header)
     */
    parseLegacyBinary() {
        if (!this.binaryData) return;

        // If we already have blocks from manifest, skip
        if (this.blocks.length > 0) return;

        const parsedBlocks = [];
        let offset = 0;

        while (offset < this.binaryData.length) {
            const view = new DataView(this.binaryData.buffer, offset);
            
            if (offset + 32 > this.binaryData.length) break;

            const size = view.getUint32(0, false);
            if (size === 0 || size > this.binaryData.length - offset) break;

            const typeBytes = this.binaryData.slice(offset + 4, offset + 8);
            const type = String.fromCharCode(...typeBytes);

            const version = view.getUint32(8, false);
            const flags = view.getUint32(12, false);

            const uuidBytes = this.binaryData.slice(offset + 16, offset + 32);
            const uuid = this.bytesToHex(uuidBytes);

            parsedBlocks.push({
                offset,
                size,
                type,
                version,
                flags,
                block_id: uuid,
                dataOffset: offset + 32,
                dataSize: size - 32,
                typeInfo: MAIFParser.BLOCK_TYPES[type] || { name: type, icon: '?', color: '#64748b' }
            });

            offset += size;
        }

        if (this.blocks.length === 0) {
            this.blocks = parsedBlocks.map((block, index) => ({
                ...block,
                index
            }));
        }
    }

    /**
     * Parse manifest structure (legacy format)
     */
    parseManifest() {
        if (!this.manifest) return;

        this.blocks = (this.manifest.blocks || []).map((block, index) => ({
            ...block,
            index,
            typeInfo: MAIFParser.BLOCK_TYPES[block.type] || { 
                name: block.type, 
                icon: '?', 
                color: '#64748b' 
            }
        }));

        if (this.manifest.signature_metadata?.provenance_chain) {
            this.provenance = this.manifest.signature_metadata.provenance_chain;
        }

        this.securityInfo = {
            public_key: this.manifest.public_key,
            signer_id: this.manifest.signature_metadata?.signer_id,
            signature: this.manifest.signature
        };

        this.fileInfo = {
            format: 'legacy',
            version: this.manifest.maif_version || this.manifest.header?.version || 'Unknown',
            created: this.manifest.created || this.manifest.header?.created_timestamp,
            agentId: this.manifest.agent_id || this.manifest.header?.agent_id,
            creatorId: this.manifest.creator_id || this.manifest.header?.creator_id,
            rootHash: this.manifest.root_hash,
            isSigned: !!this.manifest.signature
        };
    }

    /**
     * Get block data from binary
     */
    getBlockData(blockIndex) {
        if (!this.binaryData || blockIndex < 0 || blockIndex >= this.blocks.length) {
            return null;
        }

        const block = this.blocks[blockIndex];
        
        if (this.format === 'secure') {
            return this.binaryData.slice(block.dataOffset, block.dataOffset + block.dataSize);
        } else {
            const offset = block.offset || 0;
            const headerSize = 32;
            const dataOffset = offset + headerSize;
            const dataSize = block.size - headerSize;
            return this.binaryData.slice(dataOffset, dataOffset + dataSize);
        }
    }

    /**
     * Get text content from a text block
     */
    getTextContent(blockIndex) {
        const data = this.getBlockData(blockIndex);
        if (!data) return null;

        const decoder = new TextDecoder('utf-8');
        return decoder.decode(data);
    }

    /**
     * Get file overview information
     */
    getOverview() {
        if (!this.manifest && !this.fileInfo) return null;

        const blockStats = {};
        this.blocks.forEach(block => {
            const type = block.type;
            if (!blockStats[type]) {
                blockStats[type] = { count: 0, totalSize: 0 };
            }
            blockStats[type].count++;
            blockStats[type].totalSize += block.size || 0;
        });

        if (this.format === 'secure') {
            return {
                format: 'secure',
                version: this.fileInfo?.version || '2.0',
                created: this.fileInfo?.created,
                modified: this.fileInfo?.modified,
                agentId: this.fileInfo?.agentDid,
                creatorId: this.fileInfo?.creatorId,
                fileId: this.fileInfo?.fileId,
                totalBlocks: this.blocks.length,
                totalSize: this.binaryData?.length || 0,
                blockStats,
                hasSignature: this.fileInfo?.isSigned || false,
                isFinalized: this.fileInfo?.isFinalized || false,
                hasProvenance: this.provenance.length > 0,
                rootHash: this.fileInfo?.merkleRoot,
                tamperedBlocks: this.blocks.filter(b => b.isTampered).length
            };
        }

        return {
            format: 'legacy',
            version: this.fileInfo?.version || this.manifest?.maif_version || 'Unknown',
            created: this.fileInfo?.created || this.manifest?.created,
            agentId: this.fileInfo?.agentId || this.manifest?.agent_id,
            creatorId: this.fileInfo?.creatorId,
            totalBlocks: this.blocks.length,
            totalSize: this.binaryData?.length || 0,
            blockStats,
            hasSignature: !!this.manifest?.signature,
            hasProvenance: this.provenance.length > 0,
            rootHash: this.fileInfo?.rootHash || this.manifest?.root_hash
        };
    }

    /**
     * Get signature information
     */
    getSignatureInfo() {
        if (this.format === 'secure') {
            return {
                signature: this.fileInfo?.fileSignature,
                publicKey: this.securityInfo?.public_key,
                signerId: this.securityInfo?.signer_id,
                signerDid: this.securityInfo?.signer_did,
                timestamp: this.securityInfo?.signed_at,
                algorithm: this.securityInfo?.signature_algorithm || 'Ed25519',
                keyAlgorithm: this.securityInfo?.key_algorithm || 'Ed25519',
                isValid: null
            };
        }

        return {
            signature: this.manifest?.signature,
            publicKey: this.manifest?.public_key,
            signerId: this.manifest?.signature_metadata?.signer_id,
            timestamp: this.manifest?.signature_metadata?.timestamp,
            algorithm: 'Ed25519',
            isValid: null
        };
    }

    /**
     * Format bytes to hex dump
     */
    getHexDump(start = 0, length = 256) {
        if (!this.binaryData) return [];

        const lines = [];
        const end = Math.min(start + length, this.binaryData.length);

        for (let offset = start; offset < end; offset += 16) {
            const lineBytes = this.binaryData.slice(offset, Math.min(offset + 16, end));
            const hex = Array.from(lineBytes).map(b => b.toString(16).padStart(2, '0')).join(' ');
            const ascii = Array.from(lineBytes).map(b => 
                (b >= 32 && b < 127) ? String.fromCharCode(b) : '.'
            ).join('');

            lines.push({
                offset: offset.toString(16).padStart(8, '0'),
                hex: hex.padEnd(47, ' '),
                ascii
            });
        }

        return lines;
    }

    // Utility methods
    
    bytesToHex(bytes) {
        return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
    }

    readUint64(view, offset) {
        // JavaScript doesn't have native 64-bit integers, read as two 32-bit
        const high = view.getUint32(offset, false);
        const low = view.getUint32(offset + 4, false);
        // For timestamps, we can safely use Number since they fit in 53 bits
        return high * 0x100000000 + low;
    }

    nullTerminatedString(bytes) {
        const nullIndex = bytes.indexOf(0);
        const relevantBytes = nullIndex >= 0 ? bytes.slice(0, nullIndex) : bytes;
        return new TextDecoder().decode(relevantBytes);
    }

    blockTypeCodeToString(code) {
        // Convert 4-byte code to ASCII string
        const chars = [
            (code >> 24) & 0xFF,
            (code >> 16) & 0xFF,
            (code >> 8) & 0xFF,
            code & 0xFF
        ];
        return String.fromCharCode(...chars);
    }

    static formatTimestamp(timestamp) {
        if (!timestamp) return 'Unknown';
        // Handle microseconds (secure format) vs seconds (legacy)
        const ms = timestamp > 1e12 ? timestamp / 1000 : timestamp * 1000;
        const date = new Date(ms);
        return date.toLocaleString();
    }

    static formatSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    getBlockVersionHistory(blockId) {
        if (!this.manifest?.version_history) return [];
        return this.manifest.version_history[blockId] || [];
    }

    exportAsJSON() {
        return {
            format: this.format,
            overview: this.getOverview(),
            blocks: this.blocks,
            provenance: this.provenance,
            security: this.getSignatureInfo(),
            fileInfo: this.fileInfo
        };
    }
}

// Sample data for demo
const SAMPLE_MANIFEST = {
    "maif_version": "2.0",
    "format": "secure",
    "created": Date.now() * 1000,
    "agent_id": "did:maif:sample-agent-001",
    "blocks": [
        {
            "type": "TEXT",
            "offset": 444,
            "size": 500,
            "hash": "a1b2c3d4e5f6789012345678901234567890123456789012345678901234abcd",
            "version": 1,
            "block_id": "block-001",
            "is_signed": true,
            "metadata": { "source": "sample.txt" }
        },
        {
            "type": "EMBD",
            "offset": 944,
            "size": 4500,
            "hash": "b2c3d4e5f6789012345678901234567890123456789012345678901234abcde",
            "version": 1,
            "block_id": "block-002",
            "is_signed": true,
            "metadata": { "model": "all-MiniLM-L6-v2", "dimensions": 384 }
        }
    ],
    "root_hash": "sha256:merklerootabcdef1234567890",
    "signature": "SampleFileSignatureBase64==",
    "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIjAN...\n-----END PUBLIC KEY-----\n",
    "signature_metadata": {
        "signer_id": "sample-agent-001",
        "signer_did": "did:maif:sample-agent-001",
        "timestamp": Date.now() * 1000,
        "provenance_chain": [
            {
                "timestamp": Date.now() * 1000 - 100000,
                "agent_id": "sample-agent-001",
                "agent_did": "did:maif:sample-agent-001",
                "action": "genesis",
                "block_hash": "genesis_hash_12345",
                "signature": "GenesisSignature==",
                "entry_hash": "entry_hash_genesis",
                "chain_position": 0
            },
            {
                "timestamp": Date.now() * 1000 - 50000,
                "agent_id": "sample-agent-001",
                "agent_did": "did:maif:sample-agent-001",
                "action": "add_text_block",
                "block_hash": "block-001",
                "signature": "BlockSignature1==",
                "previous_entry_hash": "entry_hash_genesis",
                "entry_hash": "entry_hash_001",
                "chain_position": 1
            },
            {
                "timestamp": Date.now() * 1000 - 25000,
                "agent_id": "sample-agent-001",
                "agent_did": "did:maif:sample-agent-001",
                "action": "add_embeddings_block",
                "block_hash": "block-002",
                "signature": "BlockSignature2==",
                "previous_entry_hash": "entry_hash_001",
                "entry_hash": "entry_hash_002",
                "chain_position": 2
            },
            {
                "timestamp": Date.now() * 1000,
                "agent_id": "sample-agent-001",
                "agent_did": "did:maif:sample-agent-001",
                "action": "finalize",
                "block_hash": "merkle_root_hash",
                "signature": "FinalizeSignature==",
                "previous_entry_hash": "entry_hash_002",
                "entry_hash": "entry_hash_003",
                "chain_position": 3
            }
        ]
    }
};

function generateSampleBinary() {
    const encoder = new TextEncoder();
    const text = "This is sample AI-generated content stored in the secure MAIF format.";
    const textBytes = encoder.encode(text);
    
    const totalSize = textBytes.length + 32;
    const buffer = new ArrayBuffer(totalSize + 128);
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    
    view.setUint32(0, totalSize, false);
    bytes.set([0x54, 0x45, 0x58, 0x54], 4);
    view.setUint32(8, 1, false);
    view.setUint32(12, 0, false);
    bytes.set(textBytes, 32);
    
    return new Uint8Array(buffer);
}

window.MAIFParser = MAIFParser;
window.SAMPLE_MANIFEST = SAMPLE_MANIFEST;
window.generateSampleBinary = generateSampleBinary;
