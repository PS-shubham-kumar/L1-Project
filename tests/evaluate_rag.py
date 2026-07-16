import os
import sys
import pandas as pd
from datasets import Dataset

# Setup paths so that project modules can be loaded properly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from app import _build_chain
from llm.llm_client import get_llm
from embeddings.embedding_model import get_embedding_model


# Define golden test cases representing queries over the Indian Constitution
GOLDEN_TEST_CASES = [
    {
        "question": "what does article 14 of constitution states?",
        "ground_truth": "Article 14 guarantees equality before the law and equal protection of the laws to any person within the territory of India."
    },
    {
        "question": "What is the minimum age to be elected as the President of India?",
        "ground_truth": "According to Article 58, the minimum age required to qualify for election as President of India is 35 years."
    },
    {
        "question": "Under which article is the right to life and personal liberty protected?",
        "ground_truth": "The right to life and personal liberty is protected under Article 21 of the Indian Constitution."
    },
    {
        "question": "What are the fundamental rights?",
        "ground_truth": "The fundamental rights in the Indian Constitution include the right to equality, right to freedom of speech and expression, right to freedom of religion, right to constitutional remedies, and right against exploitation."
    },
    {
        "question": "What are the fundamental duties of a citizen of India under Article 51A?",
        "ground_truth": "Article 51A specifies eleven Fundamental Duties for citizens of India, which include respecting the national flag and constitution, safeguarding public property, and developing scientific temper."
    }
]


def run_evaluation():
    print("==================================================")
    print("   L1-PROJECT RAG PIPELINE EVALUATION (RAGAS)     ")
    print("==================================================")
    
    # 1. Initialize RAG pipeline
    try:
        print("[1/4] Initializing RAG pipeline components...")
        chain = _build_chain()
    except Exception as exc:
        print(f"Error initializing RAG chain: {exc}")
        print("Please check your .env configuration and NVIDIA_API_KEY.")
        sys.exit(1)

    # 2. Gather outputs from the active RAG chain
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("[2/4] Generating answers for golden test dataset...")
    for idx, case in enumerate(GOLDEN_TEST_CASES, 1):
        q = case["question"]
        print(f"      [{idx}/{len(GOLDEN_TEST_CASES)}] Question: '{q}'")
        try:
            res = chain.invoke(q)
            questions.append(q)
            answers.append(res["answer"])
            # Ragas expects contexts to be a list of strings
            contexts.append([doc.page_content for doc in res["source_docs"]])
            ground_truths.append(case["ground_truth"])
        except Exception as exc:
            print(f"      Error generating answer for case {idx}: {exc}")

    if not questions:
        print("No test cases were successfully run. Exiting.")
        sys.exit(1)

    # 3. Format dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # 4. Wrap NVIDIA NIM clients in Ragas wrappers
    print("[3/4] Initializing LLM and Embedding evaluators via NVIDIA NIM...")
    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embedding_model())

    # 5. Evaluate metrics
    print("[4/4] Evaluating metrics using Ragas...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
            llm=llm,
            embeddings=embeddings
        )
    except Exception as exc:
        print(f"Ragas evaluation failed: {exc}")
        sys.exit(1)

    # 6. Save and print report
    print("\n==================================================")
    print("               EVALUATION REPORT                  ")
    print("==================================================")
    
    # Ragas evaluate output has a .to_pandas() method
    df = result.to_pandas()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df[["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
    
    # Calculate averages
    print("\n--- Average Scores ---")
    for metric_name, score in result.items():
        print(f"  {metric_name.capitalize()}: {score:.4f}")
    
    report_path = os.path.join(PROJECT_ROOT, "tests", "evaluation_report.csv")
    df.to_csv(report_path, index=False)
    print(f"\nDetailed report saved to: {report_path}")
    print("==================================================")


if __name__ == "__main__":
    run_evaluation()
