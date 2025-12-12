#!/usr/bin/env python3
"""
Create knowledge base artifacts for the CrewAI enhanced RAG demo.
"""

from pathlib import Path
from typing import List, Dict

from vector_db import get_vector_db
from maif_utils import KBManager


SAMPLE_DOCS = [
    {
        "doc_id": "climate_overview",
        "title": "Climate Change Overview",
        "author": "Research Dept",
        "chunks": [
            "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities such as burning fossil fuels.",
            "Greenhouse gases like CO2 and methane trap heat in the atmosphere, intensifying the greenhouse effect and raising global temperatures.",
        ],
    },
    {
        "doc_id": "mitigation_strategies",
        "title": "Mitigation Strategies",
        "author": "Sustainability Lab",
        "chunks": [
            "Mitigation strategies include rapid decarbonization, electrification of transport, renewable energy adoption, and efficiency improvements.",
            "Carbon capture, reforestation, and grid-scale storage complement renewables to stabilize power systems and reduce emissions.",
        ],
    },
    {
        "doc_id": "adaptation_actions",
        "title": "Adaptation Actions",
        "author": "Policy Group",
        "chunks": [
            "Adaptation focuses on resilience: flood defenses, heatwave early warning systems, climate-resilient agriculture, and water management.",
            "Urban planning with green infrastructure helps manage heat islands and stormwater while improving livability.",
        ],
    },
]


def build_kb():
    kb_manager = KBManager()
    vector_db = get_vector_db()

    kb_paths: List[str] = []

    for doc in SAMPLE_DOCS:
        doc_id = doc["doc_id"]
        texts = doc["chunks"]

        # Generate embeddings once for MAIF artifacts
        embeddings = vector_db.generate_embeddings(texts)

        # Prepare chunks with embeddings
        chunks: List[Dict] = []
        for i, text in enumerate(texts):
            chunks.append(
                {
                    "text": text,
                    "embedding": embeddings[i],
                    "metadata": {
                        "title": doc["title"],
                        "author": doc["author"],
                        "section": i,
                    },
                }
            )

        # Add to vector DB
        vector_db.add_documents(doc_id=doc_id, chunks=chunks, document_metadata={"title": doc["title"]})

        # Create MAIF KB artifact
        kb_path = kb_manager.create_kb_artifact(doc_id=doc_id, chunks=chunks, document_metadata={"title": doc["title"]})
        kb_paths.append(kb_path)
        print(f"✅ Created KB artifact: {kb_path}")

    print("\nKB creation complete.")
    print("Artifacts:")
    for path in kb_paths:
        print(f"- {path}")
    print("\nVector DB location:", vector_db.persist_directory)


if __name__ == "__main__":
    build_kb()


