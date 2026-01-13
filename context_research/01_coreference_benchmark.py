"""
Coreference Benchmark

Tests entity distinction capabilities using the Coreference Challenge Set.

This benchmark specifically targets scenarios where:
- Generic tags (<PERSON>) lose information
- Semantic surrogates ({Name_hash}) preserve entity relationships

Expected outcome: Semantic masking significantly outperforms generic masking
on questions requiring referential integrity.
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_research.coref_challenge import generate_challenge_set

try:
    from privalyse_mask import PrivalyseMasker
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    sys.exit(1)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class CorefResult:
    case_id: str
    category: str
    difficulty: str
    question: str
    ground_truth: str
    
    # Results per strategy
    unmasked_answer: str
    unmasked_correct: bool
    
    presidio_answer: str
    presidio_correct: bool
    presidio_text: str  # For debugging
    
    privalyse_answer: str
    privalyse_correct: bool
    privalyse_text: str


def apply_masking(text: str, strategy: str,
                  privalyse_masker: PrivalyseMasker,
                  presidio_analyzer: AnalyzerEngine,
                  presidio_anonymizer: AnonymizerEngine) -> tuple:
    """Apply masking strategy. Returns (masked_text, mapping)."""
    if strategy == "unmasked":
        return text, {}
    elif strategy == "presidio":
        results = presidio_analyzer.analyze(text=text, language="en")
        anonymized = presidio_anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, {}
    elif strategy == "privalyse":
        masked, mapping = privalyse_masker.mask(text)
        return masked, mapping
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def ask_llm(client: OpenAI, text: str, question: str, model: str = "gpt-4o-mini") -> str:
    """Ask the LLM a question about the text."""
    
    prompt = f"""Based on the following text, answer the question concisely.

Text:
{text}

Question: {question}

Answer with just the name or short phrase. No explanation needed."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    
    return response.choices[0].message.content.strip()


def check_answer(response: str, ground_truth: str) -> bool:
    """Check if the response contains the ground truth answer."""
    response_lower = response.lower()
    ground_truth_lower = ground_truth.lower()
    
    # Check for exact match or containment
    if ground_truth_lower in response_lower:
        return True
    
    # Check first/last name separately
    name_parts = ground_truth_lower.split()
    for part in name_parts:
        if len(part) > 2 and part in response_lower:
            return True
    
    return False


def run_coref_benchmark(seed: int = 42, 
                        model: str = "gpt-4o-mini",
                        max_cases: Optional[int] = None) -> Dict:
    """Run the coreference benchmark with real LLM evaluation."""
    
    print("=" * 70)
    print("BENCHMARK: Coreference Challenge (Entity Distinction)")
    print("=" * 70)
    
    # Check for API key - FAIL if not present
    if not OPENAI_AVAILABLE:
        print("❌ ERROR: OpenAI package not installed")
        sys.exit(1)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not set")
        print("   This benchmark requires real LLM evaluation")
        print("   Set: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    print(f"✓ Using OpenAI API (model: {model})")
    
    # Generate challenge set
    print(f"\n🔧 Generating challenge set (seed={seed})...")
    challenge_data = generate_challenge_set(seed)
    cases = challenge_data["cases"]
    
    if max_cases:
        cases = cases[:max_cases]
    
    print(f"   {len(cases)} test cases")
    
    # Initialize maskers
    print("\n🔧 Initializing maskers...")
    privalyse_masker = PrivalyseMasker(languages=["en"], model_size="lg")
    presidio_analyzer = AnalyzerEngine()
    presidio_anonymizer = AnonymizerEngine()
    
    # Run evaluation
    results = []
    strategy_scores = {
        "unmasked": {"correct": 0, "total": 0},
        "presidio": {"correct": 0, "total": 0},
        "privalyse": {"correct": 0, "total": 0}
    }
    
    print(f"\n🔬 Running evaluation ({len(cases)} cases)...")
    
    for i, case in enumerate(cases):
        print(f"   [{i+1}/{len(cases)}] {case['id']}...", end=" ")
        
        text = case["text"]
        question = case["question"]
        ground_truth = case["answer"]
        
        # Apply masking (privalyse returns mapping for unmasking)
        unmasked_text = text
        presidio_text, _ = apply_masking(text, "presidio", privalyse_masker, presidio_analyzer, presidio_anonymizer)
        privalyse_text, privalyse_mapping = apply_masking(text, "privalyse", privalyse_masker, presidio_analyzer, presidio_anonymizer)
        
        # Ask LLM
        unmasked_answer = ask_llm(client, unmasked_text, question, model)
        presidio_answer = ask_llm(client, presidio_text, question, model)
        privalyse_answer_raw = ask_llm(client, privalyse_text, question, model)
        
        # Unmask the privalyse answer before checking
        privalyse_answer_unmasked = privalyse_masker.unmask(privalyse_answer_raw, privalyse_mapping)
        
        # Check answers
        unmasked_correct = check_answer(unmasked_answer, ground_truth)
        presidio_correct = check_answer(presidio_answer, ground_truth)
        privalyse_correct = check_answer(privalyse_answer_unmasked, ground_truth)
        
        # Update scores
        strategy_scores["unmasked"]["correct"] += int(unmasked_correct)
        strategy_scores["unmasked"]["total"] += 1
        strategy_scores["presidio"]["correct"] += int(presidio_correct)
        strategy_scores["presidio"]["total"] += 1
        strategy_scores["privalyse"]["correct"] += int(privalyse_correct)
        strategy_scores["privalyse"]["total"] += 1
        
        result = CorefResult(
            case_id=case["id"],
            category=case["category"],
            difficulty=case["difficulty"],
            question=question,
            ground_truth=ground_truth,
            unmasked_answer=unmasked_answer,
            unmasked_correct=unmasked_correct,
            presidio_answer=presidio_answer,
            presidio_correct=presidio_correct,
            presidio_text=presidio_text,
            privalyse_answer=f"{privalyse_answer_raw} -> {privalyse_answer_unmasked}",
            privalyse_correct=privalyse_correct,
            privalyse_text=privalyse_text
        )
        results.append(result)
        
        # Print result indicator
        indicators = []
        indicators.append("U✓" if unmasked_correct else "U✗")
        indicators.append("P✓" if presidio_correct else "P✗")
        indicators.append("S✓" if privalyse_correct else "S✗")
        print(" ".join(indicators))
    
    # Calculate final scores
    final_scores = {}
    for strategy, scores in strategy_scores.items():
        accuracy = scores["correct"] / scores["total"] if scores["total"] > 0 else 0
        final_scores[strategy] = {
            "correct": scores["correct"],
            "total": scores["total"],
            "accuracy": round(accuracy, 4)
        }
    
    return {
        "metadata": {
            "benchmark": "coreference_challenge",
            "seed": seed,
            "model": model,
            "total_cases": len(cases),
            "timestamp": datetime.now().isoformat(),
            "evaluator": "openai"
        },
        "scores": final_scores,
        "detailed_results": [asdict(r) for r in results]
    }


def print_coref_results(results: Dict):
    """Print coreference benchmark results."""
    
    print("\n" + "=" * 70)
    print("📊 COREFERENCE BENCHMARK RESULTS")
    print("=" * 70)
    
    scores = results["scores"]
    
    print(f"""
Strategy Comparison:
  
  Strategy      | Correct | Total | Accuracy
  --------------|---------|-------|----------
  Unmasked      | {scores['unmasked']['correct']:>7} | {scores['unmasked']['total']:>5} | {scores['unmasked']['accuracy']:>7.1%}
  Presidio      | {scores['presidio']['correct']:>7} | {scores['presidio']['total']:>5} | {scores['presidio']['accuracy']:>7.1%}
  Privalyse     | {scores['privalyse']['correct']:>7} | {scores['privalyse']['total']:>5} | {scores['privalyse']['accuracy']:>7.1%}

Gap Analysis:
  Privalyse vs Presidio: {(scores['privalyse']['accuracy'] - scores['presidio']['accuracy']) * 100:+.1f}%
  Privalyse vs Unmasked: {(scores['privalyse']['accuracy'] - scores['unmasked']['accuracy']) * 100:+.1f}%
""")
    
    # Show cases where Privalyse won but Presidio lost
    detailed = results.get("detailed_results", [])
    semantic_wins = [r for r in detailed if r["privalyse_correct"] and not r["presidio_correct"]]
    
    if semantic_wins:
        print("\n🎯 Cases where Semantic Masking Won (Presidio Failed):")
        for r in semantic_wins[:3]:  # Show top 3
            print(f"\n  Case: {r['case_id']}")
            print(f"  Question: {r['question']}")
            print(f"  Ground Truth: {r['ground_truth']}")
            print(f"  Presidio Answer: {r['presidio_answer']} ❌")
            print(f"  Privalyse Answer: {r['privalyse_answer']} ✓")


def save_coref_results(results: Dict, output_dir: str):
    """Save results."""
    
    # Always save to openai/ subdirectory since this requires LLM
    output_dir = os.path.join(output_dir, "openai")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"coref_benchmark_{timestamp}.json")
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coreference Benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--output", type=str, default="../results")
    args = parser.parse_args()
    
    results = run_coref_benchmark(
        seed=args.seed,
        model=args.model,
        max_cases=args.max_cases
    )
    
    print_coref_results(results)
    save_coref_results(results, args.output)
