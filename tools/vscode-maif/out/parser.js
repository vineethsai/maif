"use strict";
/**
 * MAIF File Parser for VS Code Extension
 * Supports both legacy format (with manifest) and secure format
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MAIFParser = void 0;
class MAIFParser {
    constructor() {
        this.manifest = null;
        this.binaryData = null;
        this.blocks = [];
        this.provenance = [];
        this.fileInfo = null;
        this.securityInfo = null;
        this.format = 'unknown';
    }
    loadManifestFromString(content) {
        try {
            this.manifest = JSON.parse(content);
            this.format = 'legacy';
            this.parseManifest();
        }
        catch (error) {
            throw new Error(`Failed to parse manifest: ${error}`);
        }
    }
    loadBinaryFromBuffer(buffer) {
        this.binaryData = buffer;
        this.detectAndParseBinary();
    }
    detectAndParseBinary() {
        if (!this.binaryData || this.binaryData.length < 4)
            return;
        // Check magic number
        const magic = String.fromCharCode(...this.binaryData.slice(0, 4));
        if (magic === MAIFParser.MAGIC_HEADER) {
            this.format = 'secure';
            this.parseSecureBinary();
        }
        else {
            if (this.format !== 'legacy') {
                this.format = 'legacy';
            }
            this.parseLegacyBinary();
        }
    }
    parseSecureBinary() {
        if (!this.binaryData)
            return;
        try {
            const view = new DataView(this.binaryData.buffer);
            // Parse file header (444 bytes)
            const versionMajor = view.getUint16(4, false);
            const versionMinor = view.getUint16(6, false);
            const flags = view.getUint32(8, false);
            const created = this.readUint64(view, 12);
            const modified = this.readUint64(view, 20);
            const fileId = this.bytesToHex(this.binaryData.slice(28, 44));
            const creatorId = this.bytesToHex(this.binaryData.slice(44, 60));
            const agentDid = this.nullTerminatedString(this.binaryData.slice(60, 124));
            const blockCount = view.getUint32(124, false);
            const provenanceOffset = this.readUint64(view, 128);
            const securityOffset = this.readUint64(view, 136);
            const indexOffset = this.readUint64(view, 144);
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
            // Build manifest for compatibility
            this.buildCompatibilityManifest();
        }
        catch (err) {
            console.error('Error parsing secure format:', err);
        }
    }
    parseSecureBlock(offset, index) {
        if (!this.binaryData || offset + MAIFParser.SECURE_BLOCK_HEADER_SIZE > this.binaryData.length) {
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
        const type = this.blockTypeCodeToString(blockTypeCode);
        const dataOffset = offset + MAIFParser.SECURE_BLOCK_HEADER_SIZE;
        const dataSize = size - MAIFParser.SECURE_BLOCK_HEADER_SIZE - metadataSize;
        let metadata = undefined;
        if (metadataSize > 0 && this.binaryData) {
            try {
                const metaBytes = this.binaryData.slice(dataOffset + dataSize, dataOffset + dataSize + metadataSize);
                const metaStr = new TextDecoder().decode(metaBytes);
                metadata = JSON.parse(metaStr);
            }
            catch (e) {
                metadata = { _error: 'Failed to parse metadata' };
            }
        }
        return {
            type,
            block_type: type,
            offset,
            size,
            hash: contentHash,
            version,
            previous_hash: previousHash,
            block_id: blockId,
            timestamp,
            signature,
            dataOffset,
            dataSize,
            flags,
            metadata,
            isSigned: !!(flags & 0x01),
            isImmutable: !!(flags & 0x02),
            isTampered: !!(flags & 0x20)
        };
    }
    parseProvenanceSection(offset) {
        if (!this.binaryData || offset === 0 || offset >= this.binaryData.length) {
            this.provenance = [];
            return;
        }
        try {
            const view = new DataView(this.binaryData.buffer, offset);
            const size = view.getUint32(0, false);
            const data = this.binaryData.slice(offset + 4, offset + 4 + size);
            const jsonStr = new TextDecoder().decode(data);
            this.provenance = JSON.parse(jsonStr);
        }
        catch (e) {
            console.error('Error parsing provenance:', e);
            this.provenance = [];
        }
    }
    parseSecuritySection(offset) {
        if (!this.binaryData || offset === 0 || offset >= this.binaryData.length) {
            this.securityInfo = null;
            return;
        }
        try {
            const view = new DataView(this.binaryData.buffer, offset);
            const size = view.getUint32(0, false);
            const data = this.binaryData.slice(offset + 4, offset + 4 + size);
            const jsonStr = new TextDecoder().decode(data);
            const parsed = JSON.parse(jsonStr);
            this.securityInfo = {
                publicKey: parsed.public_key,
                signerId: parsed.signer_id,
                signerDid: parsed.signer_did,
                signedAt: parsed.signed_at,
                signatureAlgorithm: parsed.signature_algorithm,
                keyAlgorithm: parsed.key_algorithm
            };
        }
        catch (e) {
            console.error('Error parsing security section:', e);
            this.securityInfo = null;
        }
    }
    buildCompatibilityManifest() {
        this.manifest = {
            maif_version: this.fileInfo?.version || '2.0',
            format: 'secure',
            created: this.fileInfo?.created || 0,
            modified: this.fileInfo?.modified,
            agent_id: this.fileInfo?.agentDid,
            file_id: this.fileInfo?.fileId,
            blocks: this.blocks,
            merkle_root: this.fileInfo?.merkleRoot,
            signature: this.fileInfo?.fileSignature,
            public_key: this.securityInfo?.publicKey,
            signature_metadata: {
                signer_id: this.securityInfo?.signerId || '',
                signer_did: this.securityInfo?.signerDid,
                timestamp: this.securityInfo?.signedAt || 0,
                signature_algorithm: this.securityInfo?.signatureAlgorithm,
                key_algorithm: this.securityInfo?.keyAlgorithm,
                provenance_chain: this.provenance
            }
        };
    }
    parseLegacyBinary() {
        if (!this.binaryData)
            return;
        if (this.blocks.length > 0)
            return;
        const parsedBlocks = [];
        let offset = 0;
        while (offset < this.binaryData.length) {
            const view = new DataView(this.binaryData.buffer, offset);
            if (offset + 32 > this.binaryData.length)
                break;
            const size = view.getUint32(0, false);
            if (size === 0 || size > this.binaryData.length - offset)
                break;
            const typeBytes = this.binaryData.slice(offset + 4, offset + 8);
            const type = String.fromCharCode(...typeBytes);
            const version = view.getUint32(8, false);
            parsedBlocks.push({
                type,
                block_type: type,
                offset,
                size,
                hash: '',
                version,
                dataOffset: offset + 32,
                dataSize: size - 32
            });
            offset += size;
        }
        this.blocks = parsedBlocks;
    }
    parseManifest() {
        if (!this.manifest)
            return;
        this.blocks = (this.manifest.blocks || []).map((block, index) => ({
            ...block,
            index
        }));
        if (this.manifest.signature_metadata?.provenance_chain) {
            this.provenance = this.manifest.signature_metadata.provenance_chain;
        }
        this.fileInfo = {
            format: 'legacy',
            version: this.manifest.maif_version || this.manifest.header?.version || 'Unknown',
            flags: 0,
            created: this.manifest.created || this.manifest.header?.created_timestamp || 0,
            blockCount: this.blocks.length,
            isSigned: !!this.manifest.signature,
            isFinalized: false
        };
    }
    getBlockData(blockIndex) {
        if (!this.binaryData || blockIndex < 0 || blockIndex >= this.blocks.length) {
            return null;
        }
        const block = this.blocks[blockIndex];
        if (this.format === 'secure' && block.dataOffset !== undefined && block.dataSize !== undefined) {
            return this.binaryData.slice(block.dataOffset, block.dataOffset + block.dataSize);
        }
        // Legacy format
        const dataOffset = block.offset + 32;
        const dataSize = block.size - 32;
        if (dataOffset + dataSize > this.binaryData.length) {
            return null;
        }
        return this.binaryData.slice(dataOffset, dataOffset + dataSize);
    }
    getTextContent(blockIndex) {
        const data = this.getBlockData(blockIndex);
        if (!data)
            return null;
        const decoder = new TextDecoder('utf-8');
        return decoder.decode(data);
    }
    getOverview() {
        if (!this.manifest && !this.fileInfo)
            return null;
        if (this.format === 'secure' && this.fileInfo) {
            return {
                format: 'secure',
                version: this.fileInfo.version,
                created: this.fileInfo.created,
                modified: this.fileInfo.modified,
                agentId: this.fileInfo.agentDid,
                fileId: this.fileInfo.fileId,
                totalBlocks: this.blocks.length,
                totalSize: this.binaryData?.length || 0,
                hasSignature: this.fileInfo.isSigned,
                isFinalized: this.fileInfo.isFinalized,
                hasProvenance: this.provenance.length > 0,
                merkleRoot: this.fileInfo.merkleRoot,
                tamperedBlocks: this.blocks.filter(b => b.isTampered).length
            };
        }
        return {
            format: 'legacy',
            version: this.manifest?.maif_version || this.manifest?.header?.version || 'Unknown',
            created: this.manifest?.created || this.manifest?.header?.created_timestamp || 0,
            agentId: this.manifest?.agent_id || this.manifest?.header?.agent_id,
            totalBlocks: this.blocks.length,
            totalSize: this.binaryData?.length || 0,
            hasSignature: !!this.manifest?.signature,
            isFinalized: false,
            hasProvenance: this.provenance.length > 0,
            merkleRoot: this.manifest?.root_hash,
            tamperedBlocks: 0
        };
    }
    getSecurityInfo() {
        if (this.format === 'secure') {
            return {
                signature: this.fileInfo?.fileSignature,
                publicKey: this.securityInfo?.publicKey,
                signerId: this.securityInfo?.signerId,
                signerDid: this.securityInfo?.signerDid,
                timestamp: this.securityInfo?.signedAt,
                algorithm: this.securityInfo?.signatureAlgorithm || 'RSA-PSS-SHA256',
                keyAlgorithm: this.securityInfo?.keyAlgorithm || 'RSA-2048'
            };
        }
        return {
            signature: this.manifest?.signature,
            publicKey: this.manifest?.public_key,
            signerId: this.manifest?.signature_metadata?.signer_id,
            timestamp: this.manifest?.signature_metadata?.timestamp,
            algorithm: 'RSA-PSS with SHA-256'
        };
    }
    getBlockTypeInfo(type) {
        return MAIFParser.BLOCK_TYPES[type] || { name: type, icon: '?', color: '#64748b' };
    }
    getHexDump(start = 0, length = 256) {
        if (!this.binaryData)
            return [];
        const lines = [];
        const end = Math.min(start + length, this.binaryData.length);
        for (let offset = start; offset < end; offset += 16) {
            const lineBytes = this.binaryData.slice(offset, Math.min(offset + 16, end));
            const hex = Array.from(lineBytes).map(b => b.toString(16).padStart(2, '0')).join(' ');
            const ascii = Array.from(lineBytes).map(b => (b >= 32 && b < 127) ? String.fromCharCode(b) : '.').join('');
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
        const high = view.getUint32(offset, false);
        const low = view.getUint32(offset + 4, false);
        return high * 0x100000000 + low;
    }
    nullTerminatedString(bytes) {
        const nullIndex = Array.from(bytes).indexOf(0);
        const relevantBytes = nullIndex >= 0 ? bytes.slice(0, nullIndex) : bytes;
        return new TextDecoder().decode(relevantBytes);
    }
    blockTypeCodeToString(code) {
        const chars = [
            (code >> 24) & 0xFF,
            (code >> 16) & 0xFF,
            (code >> 8) & 0xFF,
            code & 0xFF
        ];
        return String.fromCharCode(...chars);
    }
    static formatTimestamp(timestamp) {
        if (!timestamp)
            return 'Unknown';
        // Handle microseconds (secure format) vs seconds (legacy)
        const ms = timestamp > 1e12 ? timestamp / 1000 : timestamp * 1000;
        return new Date(ms).toLocaleString();
    }
    static formatSize(bytes) {
        if (bytes === 0)
            return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}
exports.MAIFParser = MAIFParser;
// Secure format constants
MAIFParser.SECURE_HEADER_SIZE = 444;
MAIFParser.SECURE_BLOCK_HEADER_SIZE = 372;
MAIFParser.MAGIC_HEADER = 'MAIF';
MAIFParser.BLOCK_TYPES = {
    'TEXT': { name: 'Text', icon: 'T', color: '#3b82f6' },
    'EMBD': { name: 'Embeddings', icon: 'E', color: '#8b5cf6' },
    'KGRF': { name: 'Knowledge Graph', icon: 'K', color: '#ec4899' },
    'KNOW': { name: 'Knowledge Graph', icon: 'K', color: '#ec4899' },
    'IMAG': { name: 'Image', icon: 'I', color: '#10b981' },
    'AUDI': { name: 'Audio', icon: 'A', color: '#f59e0b' },
    'VIDE': { name: 'Video', icon: 'V', color: '#ef4444' },
    'SECU': { name: 'Security', icon: 'S', color: '#6366f1' },
    'LIFE': { name: 'Lifecycle', icon: 'L', color: '#14b8a6' },
    'COMP': { name: 'Compressed', icon: 'C', color: '#64748b' },
    'ENCR': { name: 'Encrypted', icon: 'X', color: '#f97316' },
    'BINA': { name: 'Binary', icon: 'B', color: '#64748b' },
    'META': { name: 'Metadata', icon: 'M', color: '#94a3b8' }
};
//# sourceMappingURL=parser.js.map