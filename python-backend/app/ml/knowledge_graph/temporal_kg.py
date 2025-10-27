"""Temporal Knowledge Graph for tracking time-dependent relationships."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TemporalFact(BaseModel):
    """A time-stamped knowledge graph fact."""

    head: str = Field(..., description="Head entity")
    relation: str = Field(..., description="Relation type")
    tail: str = Field(..., description="Tail entity")
    timestamp: datetime = Field(..., description="Fact timestamp")
    valid_from: Optional[datetime] = Field(None, description="Validity start time")
    valid_until: Optional[datetime] = Field(None, description="Validity end time")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="unknown", description="Data source")
    metadata: Dict[str, str] = Field(default_factory=dict)


class TemporalQuery(BaseModel):
    """Query for temporal knowledge graph."""

    head: Optional[str] = None
    relation: Optional[str] = None
    tail: Optional[str] = None
    time_point: Optional[datetime] = None
    time_range: Optional[Tuple[datetime, datetime]] = None


class TemporalKnowledgeGraph:
    """Temporal knowledge graph with time-aware reasoning."""

    def __init__(self):
        """Initialize temporal knowledge graph."""
        self.facts: List[TemporalFact] = []
        self.graph = nx.MultiDiGraph()  # Multi for multiple edges with timestamps
        self.entity_timeline: Dict[str, List[TemporalFact]] = {}
        self.relation_timeline: Dict[str, List[TemporalFact]] = {}

    def add_fact(self, fact: TemporalFact) -> None:
        """Add a temporal fact to the graph.

        Args:
            fact: Temporal fact to add
        """
        self.facts.append(fact)

        # Add to graph with timestamp as edge attribute
        self.graph.add_edge(
            fact.head,
            fact.tail,
            key=fact.timestamp.isoformat(),
            relation=fact.relation,
            timestamp=fact.timestamp,
            valid_from=fact.valid_from,
            valid_until=fact.valid_until,
            confidence=fact.confidence,
            source=fact.source,
        )

        # Update timelines
        for entity in [fact.head, fact.tail]:
            if entity not in self.entity_timeline:
                self.entity_timeline[entity] = []
            self.entity_timeline[entity].append(fact)

        if fact.relation not in self.relation_timeline:
            self.relation_timeline[fact.relation] = []
        self.relation_timeline[fact.relation].append(fact)

        logger.debug(f"Added fact: {fact.head} --[{fact.relation}]--> {fact.tail} @ {fact.timestamp}")

    def query(self, query: TemporalQuery) -> List[TemporalFact]:
        """Query the temporal knowledge graph.

        Args:
            query: Temporal query

        Returns:
            List of matching facts
        """
        results = []

        for fact in self.facts:
            # Match head
            if query.head and fact.head != query.head:
                continue

            # Match relation
            if query.relation and fact.relation != query.relation:
                continue

            # Match tail
            if query.tail and fact.tail != query.tail:
                continue

            # Match time point
            if query.time_point:
                if fact.valid_from and query.time_point < fact.valid_from:
                    continue
                if fact.valid_until and query.time_point > fact.valid_until:
                    continue

            # Match time range
            if query.time_range:
                start, end = query.time_range
                # Fact must overlap with query range
                if fact.valid_until and fact.valid_until < start:
                    continue
                if fact.valid_from and fact.valid_from > end:
                    continue

            results.append(fact)

        return results

    def get_entity_history(
        self,
        entity: str,
        relation_filter: Optional[str] = None,
    ) -> List[TemporalFact]:
        """Get chronological history of an entity.

        Args:
            entity: Entity ID
            relation_filter: Optional relation type filter

        Returns:
            Sorted list of facts involving the entity
        """
        if entity not in self.entity_timeline:
            return []

        facts = self.entity_timeline[entity]

        if relation_filter:
            facts = [f for f in facts if f.relation == relation_filter]

        # Sort by timestamp
        facts_sorted = sorted(facts, key=lambda f: f.timestamp)

        return facts_sorted

    def get_snapshot_at_time(
        self,
        time_point: datetime,
    ) -> nx.DiGraph:
        """Get knowledge graph snapshot at a specific time.

        Args:
            time_point: Time point for snapshot

        Returns:
            NetworkX graph of valid facts at that time
        """
        snapshot = nx.DiGraph()

        for fact in self.facts:
            # Check if fact is valid at time_point
            if fact.valid_from and time_point < fact.valid_from:
                continue
            if fact.valid_until and time_point > fact.valid_until:
                continue

            snapshot.add_edge(
                fact.head,
                fact.tail,
                relation=fact.relation,
                confidence=fact.confidence,
            )

        return snapshot

    def track_relationship_evolution(
        self,
        head: str,
        tail: str,
        relation: Optional[str] = None,
    ) -> List[TemporalFact]:
        """Track how a relationship between two entities evolved over time.

        Args:
            head: Head entity
            tail: Tail entity
            relation: Optional relation type

        Returns:
            Chronological list of facts
        """
        query = TemporalQuery(head=head, tail=tail, relation=relation)
        facts = self.query(query)
        return sorted(facts, key=lambda f: f.timestamp)

    def predict_future_fact(
        self,
        head: str,
        relation: str,
        current_time: datetime,
    ) -> List[Tuple[str, float]]:
        """Predict future facts based on temporal patterns.

        This is a simple implementation. In production, use temporal models
        like TComplEx, TNTComplEx, or GDELT.

        Args:
            head: Head entity
            relation: Relation type
            current_time: Current timestamp

        Returns:
            List of (predicted_tail, confidence) tuples
        """
        # Get historical facts for this (head, relation) pair
        historical = []
        for fact in self.facts:
            if fact.head == head and fact.relation == relation:
                if fact.timestamp <= current_time:
                    historical.append(fact)

        if not historical:
            return []

        # Count tail occurrences
        tail_counts: Dict[str, int] = {}
        for fact in historical:
            tail_counts[fact.tail] = tail_counts.get(fact.tail, 0) + 1

        # Normalize to probabilities
        total = sum(tail_counts.values())
        predictions = [
            (tail, count / total)
            for tail, count in tail_counts.items()
        ]

        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    def get_temporal_statistics(self) -> Dict[str, any]:
        """Get statistics about temporal knowledge graph.

        Returns:
            Dictionary with statistics
        """
        timestamps = [f.timestamp for f in self.facts]

        if not timestamps:
            return {
                "num_facts": 0,
                "num_entities": 0,
                "num_relations": 0,
            }

        return {
            "num_facts": len(self.facts),
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": len(self.relation_timeline),
            "earliest_fact": min(timestamps),
            "latest_fact": max(timestamps),
            "time_span_days": (max(timestamps) - min(timestamps)).days,
            "facts_per_relation": {
                rel: len(facts)
                for rel, facts in self.relation_timeline.items()
            },
        }

    def visualize_timeline(
        self,
        entity: str,
    ) -> List[Dict[str, any]]:
        """Create timeline visualization data for an entity.

        Args:
            entity: Entity ID

        Returns:
            List of timeline events
        """
        history = self.get_entity_history(entity)

        events = []
        for fact in history:
            events.append({
                "timestamp": fact.timestamp.isoformat(),
                "relation": fact.relation,
                "other_entity": fact.tail if fact.head == entity else fact.head,
                "direction": "outgoing" if fact.head == entity else "incoming",
                "confidence": fact.confidence,
                "source": fact.source,
            })

        return events

    def detect_temporal_patterns(
        self,
        relation: str,
        window_days: int = 30,
    ) -> Dict[str, any]:
        """Detect temporal patterns in relation occurrences.

        Args:
            relation: Relation type to analyze
            window_days: Time window size in days

        Returns:
            Dictionary with pattern analysis
        """
        if relation not in self.relation_timeline:
            return {"error": "Relation not found"}

        facts = self.relation_timeline[relation]
        timestamps = [f.timestamp for f in facts]

        if len(timestamps) < 2:
            return {"error": "Insufficient data"}

        # Sort timestamps
        timestamps.sort()

        # Compute inter-arrival times
        intervals = [
            (timestamps[i+1] - timestamps[i]).total_seconds() / 86400  # days
            for i in range(len(timestamps) - 1)
        ]

        import numpy as np
        intervals_arr = np.array(intervals)

        return {
            "relation": relation,
            "num_occurrences": len(facts),
            "mean_interval_days": float(np.mean(intervals_arr)),
            "std_interval_days": float(np.std(intervals_arr)),
            "min_interval_days": float(np.min(intervals_arr)),
            "max_interval_days": float(np.max(intervals_arr)),
            "is_periodic": float(np.std(intervals_arr)) < float(np.mean(intervals_arr)) * 0.3,
        }

    def merge_with_static_kg(
        self,
        static_triples: List[Tuple[str, str, str]],
        default_timestamp: Optional[datetime] = None,
    ) -> None:
        """Merge static knowledge graph triples into temporal KG.

        Args:
            static_triples: List of (head, relation, tail) triples
            default_timestamp: Timestamp to assign (defaults to now)
        """
        if default_timestamp is None:
            default_timestamp = datetime.now()

        for head, relation, tail in static_triples:
            fact = TemporalFact(
                head=head,
                relation=relation,
                tail=tail,
                timestamp=default_timestamp,
                source="static_kg",
            )
            self.add_fact(fact)

        logger.info(f"Merged {len(static_triples)} static triples into temporal KG")


# Singleton instance
_temporal_kg: Optional[TemporalKnowledgeGraph] = None


def get_temporal_kg() -> TemporalKnowledgeGraph:
    """Get or create singleton TemporalKnowledgeGraph instance."""
    global _temporal_kg
    if _temporal_kg is None:
        _temporal_kg = TemporalKnowledgeGraph()
    return _temporal_kg
