import json
import os
import re
import time
import random
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import argparse
import openai
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

########################################################
#
#      Please add your GPT4o API key in the config.yaml file
#
########################################################
CONFIG_PATH = Path(__file__).parent / "gpt4o_config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

openai.api_type = config["azure_openai"]["api_type"]
openai.api_key = config["azure_openai"]["api_key"]
openai.api_base = config["azure_openai"]["api_base"]
openai.api_version = config["azure_openai"]["api_version"]
DEPLOYMENT_NAME = config["azure_openai"]["deployment_name"]

PROMPT_TEXT = """
You are given two images.
Your task is to determine whether these two images share a similar underlying logic—that is, whether they form an analogical pair.

Do NOT base your judgment on visual similarity (e.g., color, shape, composition) or semantic similarity (such as both showing the same object or class). 
Images that are visually or semantically similar but do NOT share the same underlying logic should receive a very low score.

Focus ONLY on whether the two images convey the same conceptual or relational logic. 
For example, if one image shows a peach's internal structures, and the other shows a Earth's internal structures, they share the same logic and should receive a very high score.

Output a single numerical score between 0 and 10, where:
10 = very strong analogical/relational similarity (same underlying logic)
0 = no logical/relational similarity

Output only the number.
"""


def encode_image(image_path: Path) -> str:
    """Encode image to base64 data URI."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime_type};base64,{b64}"


def call_gpt4o(prompt: str, image1_path: Path, image2_path: Path, max_retries: int = 5) -> str:
    """Call GPT-4o with two images and return response."""
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": encode_image(image1_path)}},
        {"type": "image_url", "image_url": {"url": encode_image(image2_path)}}
    ]
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                engine=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}
                ],
                temperature=0.7,
            )
            time.sleep(0.5)  # to avoid rate limiting
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                raise


def parse_score(response: str) -> Optional[float]:
    """Extract score from GPT response."""
    matches = re.findall(r'\b(\d+\.?\d*)\b', response)
    if matches:
        try:
            score = float(matches[0])
            return score if 0 <= score <= 10 else None
        except ValueError:
            return None
    return None


def process_single_pair(query_id: str, query_path: Path, retrieved_path: Path, method: str, rank: int) -> Dict[str, Any]:
    """Process a single query-retrieved image pair."""
    try:
        response = call_gpt4o(PROMPT_TEXT, query_path, retrieved_path)
        score = parse_score(response)
        
        return {
            "query_id": query_id,
            "query_image": str(query_path),
            "retrieved_image": str(retrieved_path),
            "method": method,
            "rank": rank,  # Which top-k position (1, 2, 3, ...)
            "response": response,
            "score": score
        }
    except Exception as e:
        return {
            "query_id": query_id,
            "query_image": str(query_path),
            "retrieved_image": str(retrieved_path),
            "method": method,
            "rank": rank,
            "response": f"Error: {e}",
            "score": None
        }


def load_pairs_from_json(json_path: Path, base_dir: Path, methods: Optional[List[str]] = None, top_k: int = 1) -> Dict[str, List[tuple]]:
    """
    Load query-retrieved pairs from retrieved_images.json.
    
    Args:
        json_path: Path to retrieved_images.json
        base_dir: Base directory for resolving relative image paths
        methods: List of methods to evaluate (None = all methods)
        top_k: Evaluate top-1 through top-k retrieved images (e.g., top_k=3 means top-1, top-2, top-3)
    
    Returns:
        Dict mapping method name -> list of (query_id, query_path, retrieved_path, rank) tuples
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter methods if specified
    if methods:
        data = {k: v for k, v in data.items() if k in methods}
    
    pairs_by_method = {}
    
    for method, queries in data.items():
        pairs = []
        for query in queries:
            query_id = query["query_image_id"]
            query_path = base_dir / query["query_image_path"]
            
            # Get ALL top-k retrieved images (from rank 1 to top_k)
            for rank in range(1, top_k + 1):
                if query["top_k_image_paths"] and len(query["top_k_image_paths"]) >= rank:
                    retrieved_path = base_dir / query["top_k_image_paths"][rank - 1]
                    pairs.append((query_id, query_path, retrieved_path, rank))
        
        pairs_by_method[method] = pairs
    
    return pairs_by_method


def evaluate_retrieval(
    json_path: str,
    output_json: str = "gpt_retrieval_scores.json",
    methods: Optional[List[str]] = None,
    top_k: int = 1,
    workers: int = 4
):
    """
    Evaluate retrieval results using GPT-4o analogical scoring.
    
    Args:
        json_path: Path to retrieved_images.json
        output_json: Output file path
        methods: List of methods to evaluate (None = all)
        top_k: Evaluate top-1 through top-k (e.g., 3 = evaluate top-1, top-2, top-3)
        workers: Number of parallel workers
    """
    json_path = Path(json_path)
    base_dir = json_path.parent  # Images are relative to JSON location
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load pairs
    print("Loading retrieval results...")
    pairs_by_method = load_pairs_from_json(json_path, base_dir, methods, top_k)
    
    total_pairs = sum(len(pairs) for pairs in pairs_by_method.values())
    num_queries = total_pairs // top_k if top_k > 0 else total_pairs
    print(f"Methods: {list(pairs_by_method.keys())}")
    print(f"Queries per method: {num_queries // len(pairs_by_method) if pairs_by_method else 0}")
    print(f"Evaluating top-1 to top-{top_k} ({top_k} pairs per query)")
    print(f"Total pairs to evaluate: {total_pairs}")
    print(f"Using {workers} workers")
    
    # Prepare all tasks
    all_tasks = []
    for method, pairs in pairs_by_method.items():
        for query_id, query_path, retrieved_path, rank in pairs:
            all_tasks.append((query_id, query_path, retrieved_path, method, rank))
    
    # Evaluate in parallel with live progress
    all_results = []
    
    # Track running averages per method
    running_scores = {method: [] for method in pairs_by_method.keys()}
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(process_single_pair, query_id, query_path, retrieved_path, method, rank): 
            (query_id, method, rank)
            for query_id, query_path, retrieved_path, method, rank in all_tasks
        }
        
        pbar = tqdm(as_completed(future_to_task), total=len(all_tasks), desc="Evaluating")
        for future in pbar:
            result = future.result()
            all_results.append(result)
            
            # Update running average
            method = result["method"]
            if result["score"] is not None:
                running_scores[method].append(result["score"])
            
            # Build progress string showing running averages
            avg_strs = []
            for m in sorted(running_scores.keys()):
                scores = running_scores[m]
                if scores:
                    avg = sum(scores) / len(scores)
                    avg_strs.append(f"{m[:8]}:{avg:.1f}({len(scores)})")
            
            # Update progress bar description
            if avg_strs:
                pbar.set_postfix_str(" | ".join(avg_strs[:4]))  # Show up to 4 methods
            
            # Print individual result with rank
            score_str = f"{result['score']:.1f}" if result['score'] is not None else "ERR"
            rank = result["rank"]
            tqdm.write(f"[{method}] {result['query_id'][:8]}... top-{rank} → {score_str}")
    
    # Aggregate results by method
    results_by_method = {}
    for result in all_results:
        method = result["method"]
        if method not in results_by_method:
            results_by_method[method] = {"pairs": [], "scores": []}
        results_by_method[method]["pairs"].append(result)
        if result["score"] is not None:
            results_by_method[method]["scores"].append(result["score"])
    
    # Compute statistics
    method_stats = {}
    for method, data in results_by_method.items():
        scores = data["scores"]
        method_stats[method] = {
            "total_pairs": len(data["pairs"]),
            "valid_pairs": len(scores),
            "average_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "pairs": data["pairs"]
        }
    
    # Build output
    output = {
        "config": {
            "json_path": str(json_path),
            "top_k": top_k,
            "methods": list(pairs_by_method.keys())
        },
        "summary": {
            method: {
                "total_pairs": stats["total_pairs"],
                "valid_pairs": stats["valid_pairs"],
                "average_score": stats["average_score"],
                "min_score": stats["min_score"],
                "max_score": stats["max_score"]
            }
            for method, stats in method_stats.items()
        },
        "detailed_results": {
            method: stats["pairs"]
            for method, stats in method_stats.items()
        }
    }
    
    # Save results
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"{'='*70}")
    print(f"\n{'Method':<25} {'Avg Score':>12} {'Valid/Total':>15}")
    print(f"{'-'*25} {'-'*12} {'-'*15}")
    
    for method, stats in sorted(method_stats.items(), key=lambda x: x[1]["average_score"] or 0, reverse=True):
        avg = stats["average_score"]
        avg_str = f"{avg:.2f}" if avg else "N/A"
        print(f"{method:<25} {avg_str:>12} {stats['valid_pairs']:>6}/{stats['total_pairs']:<6}")
    
    print(f"\nResults saved to: {output_json}")
    print(f"{'='*70}\n")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results using GPT-4o analogical scoring"
    )
    parser.add_argument(
        "--json", "-j",
        default="retrieved_images.json",
        help="Path to retrieved_images.json"
    )
    parser.add_argument(
        "--output", "-o",
        default="gpt_retrieval_scores.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=None,
        help="Methods to evaluate (default: all). E.g., --methods dino clip relsim"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=1,
        help="Evaluate top-1 through top-k (default: 1). E.g., --top-k 3 evaluates (query,top1), (query,top2), (query,top3)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    
    args = parser.parse_args()
    
    evaluate_retrieval(
        json_path=args.json,
        output_json=args.output,
        methods=args.methods,
        top_k=args.top_k,
        workers=args.workers
    )


if __name__ == "__main__":
    main()

