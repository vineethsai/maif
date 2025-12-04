# MAIF Explorer for VS Code

A Visual Studio Code extension for viewing and exploring MAIF (Multimodal Artifact File Format) files with full support for provenance tracking, Ed25519 signatures, and binary inspection.

## Features

### 🔍 Binary File Viewer
- Custom editor for `.maif` files with hex view
- Block structure detection and visualization
- File size and structure information

### 📋 Sidebar Views
- **Overview**: File metadata, version, agent info
- **Blocks**: List all content blocks with type, size, and Ed25519 signatures
- **Provenance**: Complete embedded provenance chain visualization

### 🔐 Security Features
- View Ed25519 signatures (64 bytes per block)
- Embedded public key display
- Merkle root verification
- Provenance chain verification

### 📝 Manifest Support
- Syntax highlighting for MAIF manifest JSON files
- Auto-detection of `*_manifest.json` files
- Jump to provenance and security information

## Installation

### From VSIX (Local)
1. Build the extension:
   ```bash
   cd vscode-maif
   npm install
   npm run compile
   npx vsce package
   ```
2. Install the `.vsix` file in VS Code:
   - Open Command Palette (Cmd/Ctrl + Shift + P)
   - Run "Extensions: Install from VSIX..."
   - Select the generated `.vsix` file

### From Marketplace
Coming soon!

## Usage

### Opening MAIF Files
1. **Direct Open**: Double-click any `.maif` file to open in the binary viewer
2. **Context Menu**: Right-click on `.maif` or `*_manifest.json` files and select "MAIF: Open Explorer"
3. **Command Palette**: Run "MAIF: Open Explorer" and select files

### Commands
| Command | Description |
|---------|-------------|
| `MAIF: Open Explorer` | Open MAIF file browser |
| `MAIF: Show Provenance Chain` | Display provenance timeline |
| `MAIF: Verify Signature` | Check file signature status |
| `MAIF: Show Hex View` | Open hex dump viewer |

### Sidebar Panel
Click the MAIF icon in the Activity Bar to access:
- **Overview**: Quick file summary
- **Blocks**: Browse all content blocks
- **Provenance**: View operation history

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `maif.autoDetectManifest` | `true` | Auto-load manifest files |
| `maif.showHexPreview` | `true` | Show hex in block previews |
| `maif.maxBlockPreviewSize` | `1024` | Max preview size in bytes |

## Supported File Types

- `.maif` - MAIF binary container files
- `*_manifest.json` - MAIF manifest files
- `*-manifest.json` - MAIF manifest files (alternate naming)

## MAIF Block Types

| Type | Description | Icon |
|------|-------------|------|
| TEXT | Text content | 📝 |
| EMBD | Embeddings | 🧠 |
| KGRF | Knowledge Graph | 🕸️ |
| IMAG | Image data | 🖼️ |
| AUDI | Audio data | 🎵 |
| VIDE | Video data | 🎬 |
| SECU | Security metadata | 🔐 |
| LIFE | Lifecycle data | 📊 |

## Requirements

- VS Code 1.85.0 or higher
- No additional dependencies required

## Development

```bash
# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes
npm run watch

# Package extension
npm run package
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.


