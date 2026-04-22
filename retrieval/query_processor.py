def split_query(query):
    # Simple rule-based split
    separators = [" and ", " also ", " & ", ","]

    for sep in separators:
        if sep in query.lower():
            return [q.strip() for q in query.split(sep)]

    return [query]