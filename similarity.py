import os
from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE, enhanced_schema=False
)

@tool
def similarity_search(query: str):
    """Retrieve the most relevant content from the knowledge graph based on a text query.
    Args:
        query: A natural language question about HsxTech
        
    Returns:
        List of relevant pages with hierarchical content (Main -> Section -> SubSection)
    """

    print("-------Similarity Search-------")

    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    q_vec = resp.data[0].embedding

    top_k = 2
    # Step 1: Find top matching nodes
    matched_nodes = kg.query("""
    MATCH (m:Main)
    WITH collect({type:'Main', element_id: elementId(m), score: gds.similarity.cosine(m.embedding, $vec)}) AS main_nodes
    MATCH (s:Section)
    WITH main_nodes, collect({type:'Section', element_id: elementId(s), score: gds.similarity.cosine(s.embedding, $vec)}) AS section_nodes
    WITH main_nodes + section_nodes AS combined_nodes
    MATCH (ss:SubSection)
    WITH combined_nodes, collect({type:'SubSection', element_id: elementId(ss), score: gds.similarity.cosine(ss.embedding, $vec)}) AS sub_nodes
    WITH combined_nodes + sub_nodes AS all_nodes
    UNWIND all_nodes AS n
    RETURN n.type AS type, n.element_id AS element_id, n.score AS score
    ORDER BY n.score DESC
    LIMIT $top_k
    """, params={"vec": q_vec, "top_k": top_k})

    if not matched_nodes:
        return {"message": "No relevant content found."}

    pages = []

    for node in matched_nodes:
        if node["type"] == "Main":
            # matched node is Main
            cypher = """
            MATCH (src:Source)-[:HAS_MAIN]->(m:Main)
            WHERE elementId(m) = $element_id
            OPTIONAL MATCH (m)-[:HAS_SECTION]->(sec:Section)
            OPTIONAL MATCH (sec)-[:HAS_SUBSECTION]->(sub:SubSection)
            WITH src, m, sec, collect(DISTINCT {title: sub.title, text: sub.text}) AS subsections
            RETURN src.url AS source_url,
                   m.title AS main_title,
                   m.text AS main_text,
                   collect(DISTINCT {section_title: sec.title, section_text: sec.text, subsections: subsections}) AS sections
            """
        elif node["type"] == "Section":
            # matched node is Section → find parent Main
            cypher = """
            MATCH (m:Main)-[:HAS_SECTION]->(sec:Section)
            WHERE elementId(sec) = $element_id
            MATCH (src:Source)-[:HAS_MAIN]->(m)
            OPTIONAL MATCH (m)-[:HAS_SECTION]->(s:Section)
            OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(sub:SubSection)
            WITH src, m, s, collect(DISTINCT {title: sub.title, text: sub.text}) AS subsections
            RETURN src.url AS source_url,
                   m.title AS main_title,
                   m.text AS main_text,
                   collect(DISTINCT {section_title: s.title, section_text: s.text, subsections: subsections}) AS sections
            """
        else:  # SubSection
            # matched node is SubSection → find parent Section → parent Main
            cypher = """
            MATCH (m:Main)-[:HAS_SECTION]->(s:Section)-[:HAS_SUBSECTION]->(sub:SubSection)
            WHERE elementId(sub) = $element_id
            MATCH (src:Source)-[:HAS_MAIN]->(m)
            OPTIONAL MATCH (m)-[:HAS_SECTION]->(sec:Section)
            OPTIONAL MATCH (sec)-[:HAS_SUBSECTION]->(ss:SubSection)
            WITH src, m, sec, collect(DISTINCT {title: ss.title, text: ss.text}) AS subsections
            RETURN src.url AS source_url,
                   m.title AS main_title,
                   m.text AS main_text,
                   collect(DISTINCT {section_title: sec.title, section_text: sec.text, subsections: subsections}) AS sections
            """

        page = kg.query(cypher, params={"element_id": node["element_id"]})
        if page:
            pages.append(page[0])

    return pages
