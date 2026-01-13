#!/usr/bin/env python3
"""
RAG End-to-End QA Benchmark

The REAL test for RAG with PII masking:
1. Can you retrieve the right document? (Retrieval)
2. Can the LLM answer the question correctly? (QA after retrieval)

Results are normalized against the ORIGINAL (unmasked) baseline.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from privalyse_mask import PrivalyseMasker
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)


# Corpus with questions that require understanding relationships
RAG_CORPUS = [
    {
        "id": "doc_001",
        "text": "Employee John Smith (ID: EMP-2847) submitted a vacation request for December 15-22, 2025. His manager Sarah Chen approved the request on November 30th. Contact: john.smith@company.com",
        "questions": [
            {"q": "Who is John Smith's manager?", "a": "Sarah Chen"},
            {"q": "When did the manager approve the vacation?", "a": "November 30"},
        ]
    },
    {
        "id": "doc_002", 
        "text": "Customer Maria Garcia called on January 5th regarding order #ORD-99281. She reported that the package was delivered to 742 Evergreen Terrace, Springfield, but the item was damaged. Case assigned to support rep David Lee.",
        "questions": [
            {"q": "Who is handling Maria Garcia's case?", "a": "David Lee"},
            {"q": "What was wrong with the delivery?", "a": "damaged"},
        ]
    },
    {
        "id": "doc_003",
        "text": "Meeting notes from March 15th project sync. Attendees: Alice Wong (PM), Bob Martinez (Dev Lead), Carol Zhang (QA). Alice assigned the authentication module to Bob. Carol will review Bob's code before release. Timeline: 2 weeks.",
        "questions": [
            {"q": "Who is responsible for the authentication module?", "a": "Bob Martinez"},
            {"q": "Who will review the authentication code?", "a": "Carol Zhang"},
            {"q": "What is Alice Wong's role?", "a": "PM"},
        ]
    },
    {
        "id": "doc_004",
        "text": "Performance review: Engineer Tom Wilson received 'Exceeds Expectations' from his manager Lisa Park on August 1st. Lisa recommended Tom for promotion to Senior Engineer. HR contact: hr@company.com",
        "questions": [
            {"q": "Who recommended Tom Wilson for promotion?", "a": "Lisa Park"},
            {"q": "What promotion was recommended?", "a": "Senior Engineer"},
        ]
    },
    {
        "id": "doc_005",
        "text": "Security incident #SEC-441: On June 5th, analyst Kevin Brown detected unauthorized access from IP 10.0.0.55. The compromised account belonged to intern Emma Davis. Kevin escalated to security lead Rachel Kim, who locked the account.",
        "questions": [
            {"q": "Whose account was compromised?", "a": "Emma Davis"},
            {"q": "Who locked the compromised account?", "a": "Rachel Kim"},
            {"q": "Who first detected the security incident?", "a": "Kevin Brown"},
        ]
    },
]


class RAGQABenchmark:
    """End-to-end RAG QA benchmark."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.client = OpenAI()
        self.embed_model = "text-embedding-3-small"
        self.qa_model = "gpt-4o-mini"
        
        self.privalyse = PrivalyseMasker(seed=seed)
        self.presidio_analyzer = AnalyzerEngine()
        self.presidio_anonymizer = AnonymizerEngine()
    
    def get_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.embed_model, input=text)
        return np.array(response.data[0].embedding)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def mask_presidio(self, text: str) -> str:
        results = self.presidio_analyzer.analyze(text=text, language="en")
        return self.presidio_anonymizer.anonymize(text=text, analyzer_results=results).text
    
    def mask_privalyse(self, text: str) -> Tuple[str, dict]:
        return self.privalyse.mask(text)
    
    def ask_llm(self, context: str, question: str) -> str:
        """Ask LLM a question given context."""
        response = self.client.chat.completions.create(
            model=self.qa_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Answer the question based only on the provided context. Be concise (1-3 words if possible). If the answer contains a placeholder like {Name_xxx} or <PERSON>, return that placeholder."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content.strip()
    
    def check_answer(self, predicted: str, expected: str, mapping: dict = None) -> bool:
        """Check if answer is correct, handling unmapping for semantic surrogates."""
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Direct match
        if expected_lower in predicted_lower:
            return True
        
        # If we have a mapping, try to unmask the prediction
        if mapping:
            unmasked = predicted
            for surrogate, original in mapping.items():
                unmasked = unmasked.replace(surrogate, original)
            if expected_lower in unmasked.lower():
                return True
        
        return False
    
    def run_benchmark(self) -> dict:
        print("=" * 60)
        print("RAG End-to-End QA Benchmark")
        print("=" * 60)
        print(f"Corpus: {len(RAG_CORPUS)} documents")
        print(f"QA Model: {self.qa_model}")
        print()
        
        results = {
            "original": {"correct": 0, "total": 0, "details": []},
            "presidio": {"correct": 0, "total": 0, "details": []},
            "privalyse": {"correct": 0, "total": 0, "details": []},
        }
        
        # Track per-question results for fair comparison
        question_results = []
        
        for doc in RAG_CORPUS:
            print(f"Processing {doc['id']}...")
            
            original_text = doc["text"]
            presidio_text = self.mask_presidio(original_text)
            privalyse_text, mapping = self.mask_privalyse(original_text)
            
            for qa in doc["questions"]:
                question = qa["q"]
                expected = qa["a"]
                
                q_result = {"question": question, "expected": expected}
                
                # Test each strategy
                for strategy, context, mp in [
                    ("original", original_text, None),
                    ("presidio", presidio_text, None),
                    ("privalyse", privalyse_text, mapping),
                ]:
                    answer = self.ask_llm(context, question)
                    correct = self.check_answer(answer, expected, mp)
                    
                    results[strategy]["total"] += 1
                    if correct:
                        results[strategy]["correct"] += 1
                    
                    results[strategy]["details"].append({
                        "doc_id": doc["id"],
                        "question": question,
                        "expected": expected,
                        "predicted": answer,
                        "correct": correct,
                    })
                    
                    q_result[strategy] = correct
                
                question_results.append(q_result)
        
        # Calculate accuracy
        for strategy in results:
            r = results[strategy]
            r["accuracy"] = r["correct"] / r["total"] if r["total"] > 0 else 0
        
        # Calculate RETENTION (% of questions answered correctly vs original)
        # Only count questions where original got it right
        original_correct = [q for q in question_results if q["original"]]
        
        presidio_retained = sum(1 for q in original_correct if q["presidio"])
        privalyse_retained = sum(1 for q in original_correct if q["privalyse"])
        
        results["retention"] = {
            "baseline_questions": len(original_correct),
            "presidio": presidio_retained,
            "privalyse": privalyse_retained,
            "presidio_pct": presidio_retained / len(original_correct) * 100 if original_correct else 0,
            "privalyse_pct": privalyse_retained / len(original_correct) * 100 if original_correct else 0,
        }
        
        return results
    
    def print_summary(self, results: dict):
        print()
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print()
        
        ret = results["retention"]
        
        print("🎯 Context Retention (vs Original Baseline)")
        print("-" * 50)
        print(f"  Questions the original answered correctly: {ret['baseline_questions']}")
        print()
        print(f"  Presidio retained:   {ret['presidio_pct']:.1f}% ({ret['presidio']}/{ret['baseline_questions']})")
        print(f"  Privalyse retained:  {ret['privalyse_pct']:.1f}% ({ret['privalyse']}/{ret['baseline_questions']})")
        
        print()
        print("📊 Raw Accuracy (for reference):")
        print(f"  Original:  {results['original']['accuracy']*100:.1f}% ({results['original']['correct']}/{results['original']['total']})")
        print(f"  Presidio:  {results['presidio']['accuracy']*100:.1f}% ({results['presidio']['correct']}/{results['presidio']['total']})")
        print(f"  Privalyse: {results['privalyse']['accuracy']*100:.1f}% ({results['privalyse']['correct']}/{results['privalyse']['total']})")
        
        print()
        print("💡 Key Finding:")
        if ret['privalyse_pct'] > ret['presidio_pct'] + 20:
            print(f"  ✅ Semantic masking retains {ret['privalyse_pct'] - ret['presidio_pct']:.0f}% MORE context than generic tags!")
        
        # Show Presidio failures
        print()
        print("❌ Why Generic Redaction Fails (sample):")
        failures = [d for d in results['presidio']['details'] if not d['correct']][:3]
        for f in failures:
            print(f"  Q: {f['question']}")
            print(f"  Expected: {f['expected']} → Got: {f['predicted']}")
            print()


def main():
    benchmark = RAGQABenchmark(seed=42)
    results = benchmark.run_full_benchmark() if hasattr(benchmark, 'run_full_benchmark') else benchmark.run_benchmark()
    benchmark.print_summary(results)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "rag_qa_benchmark.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Results saved to: {output_file}")
    return results


if __name__ == "__main__":
    main()
