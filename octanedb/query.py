"""
Query engine for advanced filtering and query operations.
"""

import logging
import re
from typing import Dict, Any, List, Union, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine for advanced filtering and query operations on vector metadata.
    
    Supports:
    - Complex filter expressions
    - Range queries
    - Text search
    - Logical operators
    - Custom filter functions
    """
    
    def __init__(self):
        """Initialize the query engine."""
        self._operators = {
            "$eq": self._eq,
            "$ne": self._ne,
            "$gt": self._gt,
            "$gte": self._gte,
            "$lt": self._lt,
            "$lte": self._lte,
            "$in": self._in,
            "$nin": self._nin,
            "$regex": self._regex,
            "$exists": self._exists,
            "$and": self._and,
            "$or": self._nor,
            "$not": self._not,
            "$text": self._text_search
        }
        
        # Text search index for faster text queries
        self._text_index: Dict[str, Dict[str, List[int]]] = {}
        
        logger.info("Query engine initialized")
    
    def evaluate_filter(self, metadata: Dict[str, Any], filter_expr: Dict[str, Any]) -> bool:
        """
        Evaluate a filter expression against metadata.
        
        Args:
            metadata: Vector metadata to evaluate
            filter_expr: Filter expression
            
        Returns:
            True if metadata matches filter, False otherwise
        """
        try:
            return self._evaluate_expression(metadata, filter_expr)
        except Exception as e:
            logger.error(f"Error evaluating filter: {e}")
            return False
    
    def _evaluate_expression(self, metadata: Dict[str, Any], expr: Any) -> bool:
        """Evaluate a single expression."""
        if isinstance(expr, dict):
            # Handle operator expressions
            for operator, value in expr.items():
                if operator in self._operators:
                    return self._operators[operator](metadata, value)
                else:
                    # Direct field comparison
                    return self._eq(metadata, {operator: value})
        elif isinstance(expr, list):
            # Handle list expressions (AND logic)
            return all(self._evaluate_expression(metadata, item) for item in expr)
        else:
            # Simple value comparison
            return expr is True
        
        return False
    
    def _eq(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Equality operator."""
        if isinstance(value, dict):
            field, expected_value = next(iter(value.items()))
            return metadata.get(field) == expected_value
        return False
    
    def _ne(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Not equality operator."""
        if isinstance(value, dict):
            field, expected_value = next(iter(value.items()))
            return metadata.get(field) != expected_value
        return False
    
    def _gt(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Greater than operator."""
        if isinstance(value, dict):
            field, expected_value = next(iter(value.items()))
            field_value = metadata.get(field)
            if field_value is not None and expected_value is not None:
                return field_value > expected_value
        return False
    
    def _gte(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Greater than or equal operator."""
        if isinstance(value, dict):
            field, expected_value = next(iter(value.items()))
            field_value = metadata.get(field)
            if field_value is not None and expected_value is not None:
                return field_value >= expected_value
        return False
    
    def _lt(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Less than operator."""
        if isinstance(value, dict):
            field, expected_value = next(iter(value.items()))
            field_value = metadata.get(field)
            if field_value is not None and expected_value is not None:
                return field_value < expected_value
        return False
    
    def _lte(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Less than or equal operator."""
        if isinstance(value, dict):
            field, expected_value = next(iter(value.items()))
            field_value = metadata.get(field)
            if field_value is not None and expected_value is not None:
                return field_value <= expected_value
        return False
    
    def _in(self, metadata: Dict[str, Any], value: Any) -> bool:
        """In operator."""
        if isinstance(value, dict):
            field, expected_values = next(iter(value.items()))
            if isinstance(expected_values, list):
                field_value = metadata.get(field)
                return field_value in expected_values
        return False
    
    def _nin(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Not in operator."""
        if isinstance(value, dict):
            field, expected_values = next(iter(value.items()))
            if isinstance(expected_values, list):
                field_value = metadata.get(field)
                return field_value not in expected_values
        return False
    
    def _regex(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Regex operator."""
        if isinstance(value, dict):
            field, pattern = next(iter(value.items()))
            field_value = metadata.get(field)
            if isinstance(field_value, str) and isinstance(pattern, str):
                try:
                    return bool(re.search(pattern, field_value, re.IGNORECASE))
                except re.error:
                    return False
        return False
    
    def _exists(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Exists operator."""
        if isinstance(value, dict):
            field, should_exist = next(iter(value.items()))
            field_exists = field in metadata
            return field_exists == should_exist
        return False
    
    def _and(self, metadata: Dict[str, Any], value: Any) -> bool:
        """AND operator."""
        if isinstance(value, list):
            return all(self._evaluate_expression(metadata, item) for item in value)
        return False
    
    def _or(self, metadata: Dict[str, Any], value: Any) -> bool:
        """OR operator."""
        if isinstance(value, list):
            return any(self._evaluate_expression(metadata, item) for item in value)
        return False
    
    def _nor(self, metadata: Dict[str, Any], value: Any) -> bool:
        """NOR operator."""
        if isinstance(value, list):
            return not any(self._evaluate_expression(metadata, item) for item in value)
        return False
    
    def _not(self, metadata: Dict[str, Any], value: Any) -> bool:
        """NOT operator."""
        return not self._evaluate_expression(metadata, value)
    
    def _text_search(self, metadata: Dict[str, Any], value: Any) -> bool:
        """Text search operator."""
        if isinstance(value, dict):
            field, search_text = next(iter(value.items()))
            field_value = metadata.get(field)
            if isinstance(field_value, str) and isinstance(search_text, str):
                return search_text.lower() in field_value.lower()
        return False
    
    def build_text_index(self, collection_name: str, metadata_list: List[Dict[str, Any]], vector_ids: List[int]) -> None:
        """
        Build a text search index for faster text queries.
        
        Args:
            collection_name: Name of the collection
            metadata_list: List of metadata dictionaries
            vector_ids: Corresponding vector IDs
        """
        if collection_name not in self._text_index:
            self._text_index[collection_name] = {}
        
        # Index text fields
        for metadata, vector_id in zip(metadata_list, vector_ids):
            for field, value in metadata.items():
                if isinstance(value, str):
                    if field not in self._text_index[collection_name]:
                        self._text_index[collection_name][field] = {}
                    
                    # Create word index
                    words = value.lower().split()
                    for word in words:
                        if word not in self._text_index[collection_name][field]:
                            self._text_index[collection_name][field][word] = []
                        self._text_index[collection_name][field][word].append(vector_id)
        
        logger.info(f"Text index built for collection '{collection_name}'")
    
    def search_text(self, collection_name: str, field: str, query: str) -> List[int]:
        """
        Search text using the built index.
        
        Args:
            collection_name: Name of the collection
            field: Field to search in
            query: Search query
            
        Returns:
            List of vector IDs matching the query
        """
        if collection_name not in self._text_index:
            return []
        
        if field not in self._text_index[collection_name]:
            return []
        
        query_words = query.lower().split()
        matching_ids = set()
        
        for word in query_words:
            if word in self._text_index[collection_name][field]:
                matching_ids.update(self._text_index[collection_name][field][word])
        
        return list(matching_ids)
    
    def create_aggregation_pipeline(self, pipeline: List[Dict[str, Any]]) -> Callable:
        """
        Create an aggregation pipeline for complex data processing.
        
        Args:
            pipeline: List of aggregation stages
            
        Returns:
            Callable function that executes the pipeline
        """
        def execute_pipeline(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            result = data
            
            for stage in pipeline:
                if "$match" in stage:
                    result = [item for item in result if self.evaluate_filter(item, stage["$match"])]
                elif "$project" in stage:
                    result = self._project_fields(result, stage["$project"])
                elif "$sort" in stage:
                    result = self._sort_data(result, stage["$sort"])
                elif "$limit" in stage:
                    result = result[:stage["$limit"]]
                elif "$skip" in stage:
                    result = result[stage["$skip"]:]
                elif "$group" in stage:
                    result = self._group_data(result, stage["$group"])
                elif "$count" in stage:
                    result = [{"count": len(result)}]
            
            return result
        
        return execute_pipeline
    
    def _project_fields(self, data: List[Dict[str, Any]], projection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Project specific fields from data."""
        result = []
        for item in data:
            projected_item = {}
            for field, include in projection.items():
                if include:
                    if field in item:
                        projected_item[field] = item[field]
            result.append(projected_item)
        return result
    
    def _sort_data(self, data: List[Dict[str, Any]], sort_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sort data based on sort specification."""
        def sort_key(item):
            keys = []
            for field, direction in sort_spec.items():
                value = item.get(field, 0)
                if direction == -1:
                    value = -value if isinstance(value, (int, float)) else value
                keys.append(value)
            return tuple(keys)
        
        return sorted(data, key=sort_key)
    
    def _group_data(self, data: List[Dict[str, Any]], group_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group data based on group specification."""
        groups = {}
        
        for item in data:
            group_key = tuple(item.get(field) for field in group_spec["_id"].values())
            
            if group_key not in groups:
                groups[group_key] = {"_id": dict(zip(group_spec["_id"].keys(), group_key))}
            
            # Apply aggregations
            for field, agg in group_spec.items():
                if field == "_id":
                    continue
                
                if agg["$sum"] in item:
                    if field not in groups[group_key]:
                        groups[group_key][field] = 0
                    groups[group_key][field] += item[agg["$sum"]]
                elif agg["$avg"] in item:
                    if field not in groups[group_key]:
                        groups[group_key][field] = {"sum": 0, "count": 0}
                    groups[group_key][field]["sum"] += item[agg["$avg"]]
                    groups[group_key][field]["count"] += 1
                elif agg["$min"] in item:
                    if field not in groups[group_key]:
                        groups[group_key][field] = float('inf')
                    groups[group_key][field] = min(groups[group_key][field], item[agg["$min"]])
                elif agg["$max"] in item:
                    if field not in groups[group_key]:
                        groups[group_key][field] = float('-inf')
                    groups[group_key][field] = max(groups[group_key][field], item[agg["$max"]])
        
        # Convert averages
        for group in groups.values():
            for field, value in group.items():
                if isinstance(value, dict) and "sum" in value and "count" in value:
                    group[field] = value["sum"] / value["count"] if value["count"] > 0 else 0
        
        return list(groups.values())
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        stats = {
            "text_indexes": len(self._text_index),
            "total_indexed_fields": sum(len(fields) for fields in self._text_index.values()),
            "total_indexed_words": sum(
                sum(len(words) for words in fields.values())
                for fields in self._text_index.values()
            )
        }
        
        return stats
    
    def clear_text_index(self, collection_name: str = None) -> None:
        """Clear text search index."""
        if collection_name:
            if collection_name in self._text_index:
                del self._text_index[collection_name]
                logger.info(f"Text index cleared for collection '{collection_name}'")
        else:
            self._text_index.clear()
            logger.info("All text indexes cleared")
