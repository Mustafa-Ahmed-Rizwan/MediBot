def format_response(result):
    """
    Format the LLM response for display in Chainlit with markdown.
    """
    if not result.strip():
        return "No response generated."
    
    # Ensure the response is clean and formatted as markdown
    return f"**Answer:** {result}"

def format_sources(source_documents):
    """
    Format source documents in a Grok-like style: numbered list with metadata and content snippet.
    """
    if not source_documents:
        return "**Sources:** None"

    sources_text = "**Sources:**\n\n"
    for i, doc in enumerate(source_documents, 1):
        meta = doc.metadata
        # Build metadata display
        metadata_parts = []
        if "source" in meta:
            metadata_parts.append(f"**Source**: {meta['source']}")
        if "title" in meta:
            metadata_parts.append(f"**Title**: {meta['title']}")
        if "page" in meta:
            metadata_parts.append(f"**Page**: {meta['page']}")
        
        metadata_text = " | ".join(metadata_parts) if metadata_parts else "No metadata available"
        #bones
        content_snippet = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
        
        # Grok-like formatting: numbered, with metadata and snippet
        sources_text += f"**{i}. {metadata_text}**\n" \
                        f"> {content_snippet}\n\n"
    
    return sources_text.rstrip()