# MAIF Explorer

A beautiful, interactive web-based explorer for viewing and understanding MAIF (Multimodal Artifact File Format) files. Explore provenance chains, Ed25519 signatures, block structures, and binary content.

![MAIF Explorer Screenshot](screenshot.png)

## Features

### 📊 Overview Dashboard
- File metadata and version information
- Ed25519 signature status and verification
- Block statistics with type breakdown
- Merkle root integrity check

### 📦 Block Explorer
- Visual grid of all content blocks
- Search and filter by type
- Detailed block information with signatures
- Content preview for text blocks

### 🔗 Provenance Timeline
- Complete embedded operation history
- Genesis block tracking
- Agent identification (DID support)
- Ed25519 signatures per entry (64 bytes)

### 🔐 Security Panel
- Embedded public key display (32 bytes)
- Ed25519 signature verification
- Signer information
- Merkle root verification

### 🔢 Binary Viewer
- Hex dump with ASCII preview
- Offset navigation
- Copy to clipboard
- Block boundary highlighting

## Usage

### Local Usage
Simply open `index.html` in a modern web browser:

```bash
# macOS
open tools/maif-explorer/index.html

# Linux
xdg-open tools/maif-explorer/index.html

# Windows
start tools/maif-explorer/index.html
```

### With a Local Server (Recommended)
For better file handling, serve via HTTP:

```bash
# Python 3
python3 -m http.server 8080

# Node.js (npx)
npx serve .
```

Then open `http://localhost:8080` in your browser.

### Loading MAIF Files

1. **Drag & Drop**: Drag your `.maif` and `*_manifest.json` files directly onto the drop zone
2. **Browse**: Click "Browse Files" to select files
3. **Sample**: Click "Load Sample" to explore with demo data

## File Requirements

The explorer works best with both files:
- **`.maif`** - Binary container file
- **`*_manifest.json`** - Accompanying manifest with metadata

You can load either file individually, but the full experience requires both.

## Interface Guide

### Tab Navigation
- **Overview**: High-level file summary and statistics
- **Blocks**: Browse and search content blocks
- **Provenance**: View the complete operation timeline
- **Security**: Examine signatures and keys
- **Binary**: Inspect raw hex data

### Theme Support
Click the sun/moon icon in the header to toggle between dark and light themes. Your preference is saved locally.

### Block Details
Click any block card to open a detailed side panel showing:
- Block ID and type
- Size and offset information
- Version and hash values
- Metadata (if present)
- Content preview (for text blocks)

## Technical Details

### Supported Block Types
| Code | Type | Description |
|------|------|-------------|
| TEXT | Text | UTF-8 text content |
| EMBD | Embeddings | Vector embeddings |
| KGRF | Knowledge Graph | Graph data |
| IMAG | Image | Image data |
| AUDI | Audio | Audio data |
| VIDE | Video | Video data |
| SECU | Security | Security metadata |
| LIFE | Lifecycle | Lifecycle data |

### Binary Format
The explorer parses the secure MAIF binary format with:
- 252-byte file header (includes Ed25519 public key and Merkle root)
- 180-byte block headers (includes 64-byte Ed25519 signatures)
- Big-endian byte ordering
- SHA-256 content hashing

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Customization

### Styling
Edit `styles.css` to customize:
- Color scheme (CSS variables in `:root`)
- Font families
- Spacing and sizing

### Extending
The parser (`maif-parser.js`) is modular and can be extended:

```javascript
// Add custom block type
MAIFParser.BLOCK_TYPES['CUSTOM'] = {
    name: 'Custom Type',
    icon: 'C',
    color: '#ff6b6b'
};
```

## Files

```
maif-explorer/
├── index.html        # Main HTML structure
├── styles.css        # Styling and themes
├── maif-parser.js    # MAIF format parser
├── app.js            # Application logic
└── README.md         # This file
```

## Development

No build step required! Just edit the files directly. For development:

1. Make changes to any file
2. Refresh the browser
3. Changes appear immediately

## License

MIT License - See the main project LICENSE file.

## Related

- [MAIF Python Library](../../maif/) - Python implementation
- [VS Code Extension](../vscode-maif/) - VS Code integration
- [MAIF Documentation](../../docs/) - Full documentation


