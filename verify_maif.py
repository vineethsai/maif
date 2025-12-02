

from maif.core import MAIFEncoder, MAIFDecoder
import os
import json

def test_basic_operations():
    print("Testing basic MAIF operations...")
    
    maif_file = "agent_data.maif"
    manifest_file = "agent_data.manifest.json"
    
    # Clean up previous run
    if os.path.exists(maif_file): os.remove(maif_file)
    if os.path.exists(manifest_file): os.remove(manifest_file)

    try:
        # Create MAIF file
        print("Creating MAIF file...")
        encoder = MAIFEncoder()
        encoder.add_text_block(
            text="Agent conversation data",
            metadata={"agent_id": "agent-001", "timestamp": 1234567890}
        )
        
        # Save to file
        encoder.save(maif_file, manifest_file)
        print(f"Saved {maif_file} and {manifest_file}")

        # Read MAIF file
        print(f"Reading {maif_file}...")
        decoder = MAIFDecoder(maif_file, manifest_file)
        
        count = 0
        # Access blocks directly from decoder.blocks
        for block in decoder.blocks:
            print(f"Block {count}: Type: {block.block_type}, Size: {block.size}")
            # Get data using helper method
            data = decoder._extract_block_data(block)
            print(f"  Data: {data}")
            print(f"  Metadata: {block.metadata}")
            count += 1
            
        print(f"Successfully read {count} blocks.")
        
        # Verify text blocks specifically
        text_blocks = decoder.get_text_blocks()

        print(f"Text blocks found: {len(text_blocks)}")
        for tb in text_blocks:
            if isinstance(tb, dict):
                print(f"  Content: {tb.get('text', tb)}")
            else:
                print(f"  Content: {tb}")


    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Clean up
    if os.path.exists(maif_file):
        os.remove(maif_file)
    if os.path.exists(manifest_file):
        os.remove(manifest_file)

if __name__ == "__main__":
    test_basic_operations()
