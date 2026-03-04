import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .retriever import get_retriever_for_query, DEFAULT_CATEGORY
from .llm import explain_chunks

@csrf_exempt
def query_easa(request) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    query = data.get("query", "").strip()
    if not query:
        return JsonResponse({"error": "Missing 'query' field"}, status=400)

    category = data.get("category", DEFAULT_CATEGORY)
    mode = data.get("mode", "raw").lower()
    format_type = data.get("format", "md").lower()

    retriever, lang_used = get_retriever_for_query(query, category=category)
    results = retriever.get_relevant_documents(query)

    chunks = [
        {
            "text": r.page_content,
            "metadata": r.metadata
        }
        for r in results
    ]

    match mode:
        case "source": # raw chunks frm the vectorstore
            return JsonResponse({
                "mode": "source",
                "category": category,
                "language": lang_used,
                "results": chunks
            })

        case "answer": # LLM elaborated explanation of the raw chunks
            explanation = explain_chunks(chunks, query, format_type)
            return JsonResponse({
                "mode": "answer",
                "category": category,
                "language": lang_used,
                "explanation": explanation
            })

        case "full": # source + answer for auditing and debugging
            explanation = explain_chunks(chunks, query, format_type)
            return JsonResponse({
                "mode": "full",
                "category": category,
                "language": lang_used,
                "results": chunks,
                "explanation": explanation
            })

        case _: # anything else the user thought they could have but actually cannot
            return JsonResponse(
                {"error": f"Unknown mode '{mode}'. Use source, answer, or full."},
                status=400
            )

