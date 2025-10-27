"""Neo4j service for knowledge graph operations."""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from neo4j import GraphDatabase
from dotenv import load_dotenv

from app.logging_conf import get_logger

logger = get_logger(__name__)

# Load Neo4j credentials
load_dotenv('.env.neo4j')


class Neo4jService:
    """Service for interacting with Neo4j graph database."""

    def __init__(self):
        """Initialize Neo4j connection."""
        self.uri = os.getenv('NEO4J_URI', 'neo4j+s://dac60a90.databases.neo4j.io')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')

        if not self.password:
            logger.warning("Neo4j password not found in environment")
            self.driver = None
        else:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                # Test connection
                self.driver.verify_connectivity()
                logger.info(f"Neo4j connected successfully to {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self.driver = None

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def create_drug_node(self, drug_id: str, properties: Dict[str, Any]) -> bool:
        """Create a drug node in Neo4j."""
        if not self.driver:
            logger.warning("Neo4j driver not initialized")
            return False

        query = """
        MERGE (d:Drug {id: $drug_id})
        SET d += $properties
        RETURN d
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, drug_id=drug_id, properties=properties)
                result.single()
                logger.info(f"Created/updated drug node: {drug_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to create drug node: {e}")
            return False

    def create_disease_node(self, disease_id: str, properties: Dict[str, Any]) -> bool:
        """Create a disease node in Neo4j."""
        if not self.driver:
            return False

        query = """
        MERGE (d:Disease {id: $disease_id})
        SET d += $properties
        RETURN d
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, disease_id=disease_id, properties=properties)
                result.single()
                logger.info(f"Created/updated disease node: {disease_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to create disease node: {e}")
            return False

    def create_treats_relationship(
        self,
        drug_id: str,
        disease_id: str,
        score: float,
        evidence: Dict[str, Any]
    ) -> bool:
        """Create a TREATS relationship between drug and disease."""
        if not self.driver:
            return False

        # Flatten evidence dict to Neo4j-compatible properties
        evidence_props = {}
        if 'confidence' in evidence:
            evidence_props['confidence'] = evidence['confidence']
        if 'evidence_sources' in evidence and isinstance(evidence['evidence_sources'], list):
            evidence_props['evidence_sources'] = evidence['evidence_sources']

        query = """
        MATCH (drug:Drug {id: $drug_id})
        MATCH (disease:Disease {id: $disease_id})
        MERGE (drug)-[r:TREATS]->(disease)
        SET r.score = $score,
            r.confidence = $confidence,
            r.evidence_sources = $evidence_sources,
            r.updated = datetime()
        RETURN r
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    drug_id=drug_id,
                    disease_id=disease_id,
                    score=score,
                    confidence=evidence_props.get('confidence', 0.0),
                    evidence_sources=evidence_props.get('evidence_sources', [])
                )
                result.single()
                logger.info(f"Created TREATS relationship: {drug_id} -> {disease_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to create TREATS relationship: {e}")
            return False

    def find_paths(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 4
    ) -> List[Dict[str, Any]]:
        """Find paths between two nodes."""
        if not self.driver:
            return []

        query = """
        MATCH path = shortestPath(
            (start {id: $start_id})-[*1..$max_hops]-(end {id: $end_id})
        )
        RETURN path,
               length(path) as path_length,
               [n in nodes(path) | labels(n)[0]] as node_types,
               [n in nodes(path) | n.id] as node_ids,
               [r in relationships(path) | type(r)] as rel_types
        LIMIT 10
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    start_id=start_id,
                    end_id=end_id,
                    max_hops=max_hops
                )

                paths = []
                for record in result:
                    paths.append({
                        'path_length': record['path_length'],
                        'node_types': record['node_types'],
                        'node_ids': record['node_ids'],
                        'rel_types': record['rel_types']
                    })

                logger.info(f"Found {len(paths)} paths between {start_id} and {end_id}")
                return paths

        except Exception as e:
            logger.error(f"Failed to find paths: {e}")
            return []

    def get_drug_connections(
        self,
        drug_id: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get all connections for a drug."""
        if not self.driver:
            return {'diseases': [], 'targets': [], 'pathways': []}

        query = """
        MATCH (d:Drug {id: $drug_id})-[r]->(connected)
        RETURN type(r) as rel_type,
               labels(connected)[0] as connected_type,
               connected.id as connected_id,
               connected.name as connected_name,
               r as relationship
        LIMIT $limit
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, drug_id=drug_id, limit=limit)

                connections = {
                    'diseases': [],
                    'targets': [],
                    'pathways': [],
                    'other': []
                }

                for record in result:
                    conn_data = {
                        'id': record['connected_id'],
                        'name': record.get('connected_name', record['connected_id']),
                        'relationship': record['rel_type']
                    }

                    conn_type = record['connected_type'].lower()
                    if conn_type == 'disease':
                        connections['diseases'].append(conn_data)
                    elif conn_type in ['protein', 'gene', 'target']:
                        connections['targets'].append(conn_data)
                    elif conn_type == 'pathway':
                        connections['pathways'].append(conn_data)
                    else:
                        connections['other'].append(conn_data)

                return connections

        except Exception as e:
            logger.error(f"Failed to get drug connections: {e}")
            return {'diseases': [], 'targets': [], 'pathways': []}

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self.driver:
            return {'error': 'Neo4j not connected'}

        query = """
        MATCH (n)
        WITH labels(n)[0] as label, count(*) as count
        RETURN label, count
        ORDER BY count DESC
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)

                stats = {
                    'node_counts': {},
                    'total_nodes': 0
                }

                for record in result:
                    label = record['label']
                    count = record['count']
                    stats['node_counts'][label] = count
                    stats['total_nodes'] += count

                # Get relationship counts
                rel_query = """
                MATCH ()-[r]->()
                WITH type(r) as rel_type, count(*) as count
                RETURN rel_type, count
                ORDER BY count DESC
                """

                result = session.run(rel_query)
                stats['relationship_counts'] = {}
                stats['total_relationships'] = 0

                for record in result:
                    rel_type = record['rel_type']
                    count = record['count']
                    stats['relationship_counts'][rel_type] = count
                    stats['total_relationships'] += count

                return stats

        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            return {'error': str(e)}

    def populate_from_predictions(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Populate Neo4j from prediction results."""
        if not self.driver:
            return {'drugs': 0, 'diseases': 0, 'relationships': 0}

        drugs_created = 0
        diseases_created = 0
        rels_created = 0

        for pred in predictions:
            drug_id = pred.get('drug_id')
            disease_id = pred.get('disease_id')
            score = pred.get('score', 0.0)

            if not drug_id or not disease_id:
                continue

            # Create drug node
            if self.create_drug_node(drug_id, {
                'name': pred.get('drug_name', drug_id),
                'chembl_id': drug_id
            }):
                drugs_created += 1

            # Create disease node
            if self.create_disease_node(disease_id, {
                'name': pred.get('disease_name', disease_id)
            }):
                diseases_created += 1

            # Create relationship
            if self.create_treats_relationship(
                drug_id,
                disease_id,
                score,
                {
                    'confidence': pred.get('confidence', 0.0),
                    'evidence_sources': pred.get('evidence_sources', [])
                }
            ):
                rels_created += 1

        logger.info(
            f"Populated Neo4j: {drugs_created} drugs, "
            f"{diseases_created} diseases, {rels_created} relationships"
        )

        return {
            'drugs': drugs_created,
            'diseases': diseases_created,
            'relationships': rels_created
        }

    def clear_all_data(self) -> bool:
        """Clear all data from Neo4j (use with caution!)."""
        if not self.driver:
            return False

        query = "MATCH (n) DETACH DELETE n"

        try:
            with self.driver.session(database=self.database) as session:
                session.run(query)
                logger.warning("Cleared all data from Neo4j")
                return True
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False


# Singleton instance
_neo4j_service: Optional[Neo4jService] = None


def get_neo4j_service() -> Neo4jService:
    """Get or create Neo4j service instance."""
    global _neo4j_service

    if _neo4j_service is None:
        _neo4j_service = Neo4jService()

    return _neo4j_service
