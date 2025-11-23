"""Download SQuAD dataset and convert to markdown files"""
from datasets import load_dataset
from pathlib import Path
import json

def main():
    print("📦 Downloading SQuAD dataset...")

    # Load SQuAD (100 examples for start)
    dataset = load_dataset("squad", split="train[:100]")

    # Create directories (absolute paths from project root)
    project_root = Path(__file__).parent.parent  # scripts/ -> project root
    docs_dir = project_root / "data" / "documents" / "squad"
    golden_dir = project_root / "data" / "golden_set"

    docs_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    print(f"📝 Processing {len(dataset)} examples...")

    # Convert to .md and extract Q&A
    golden_set = []

    for idx, example in enumerate(dataset):
        # Save context as .md
        doc_path = docs_dir / f"doc_{idx:03d}.md"

        # Add title and context
        content = f"# Document {idx}\n\n{example['context']}"
        doc_path.write_text(content, encoding='utf-8')

        # Save Q&A pair
        golden_set.append({
            "question": example["question"],
            "answer": example["answers"]["text"][0],
            "document": f"doc_{idx:03d}.md",
            "context": example["context"][:200] + "..."  # Preview
        })

    # Save golden set
    golden_path = golden_dir / "squad_qa.json"
    with open(golden_path, 'w', encoding='utf-8') as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)

    print(f"✅ Created {len(dataset)} documents in {docs_dir}")
    print(f"✅ Created golden set with {len(golden_set)} Q&A pairs in {golden_path}")
    print("\n📊 Example Q&A:")
    print(f"Q: {golden_set[0]['question']}")
    print(f"A: {golden_set[0]['answer']}")

if __name__ == "__main__":
    main()