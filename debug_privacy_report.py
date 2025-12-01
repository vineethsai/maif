
import sys
import os
from maif.core import MAIFEncoder
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode

def debug_privacy_report():
    print("Starting debug_privacy_report")
    privacy_engine = PrivacyEngine()
    encoder = MAIFEncoder(
        agent_id="test_agent",
        privacy_engine=privacy_engine,
        enable_privacy=True
    )
    
    print(f"Initial blocks: {len(encoder.blocks)}")
    
    encoder.add_text_block("Public information")
    print(f"After block 1: {len(encoder.blocks)}")
    
    encoder.add_text_block(
        "Confidential data",
        privacy_level=PrivacyLevel.HIGH,
        encryption_mode=EncryptionMode.AES_GCM
    )
    print(f"After block 2: {len(encoder.blocks)}")
    
    report = encoder.get_privacy_report()
    print(f"Report: {report}")
    
    if report["total_blocks"] != 2:
        print("FAILURE: total_blocks != 2")
    else:
        print("SUCCESS")

if __name__ == "__main__":
    debug_privacy_report()
