"""Query optimization for storage backends."""
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

from sqlalchemy import text, inspect, Index
from sqlalchemy.sql import Select
from sqlalchemy.orm import Query

from ..core.logging import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of queries for optimization."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    JOIN = "join"


@dataclass
class QueryPlan:
    """Execution plan for a query."""
    query_type: QueryType
    estimated_cost: float
    estimated_rows: int
    uses_index: bool
    index_names: List[str] = field(default_factory=list)
    optimization_hints: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_type': self.query_type.value,
            'estimated_cost': self.estimated_cost,
            'estimated_rows': self.estimated_rows,
            'uses_index': self.uses_index,
            'index_names': self.index_names,
            'optimization_hints': self.optimization_hints,
            'execution_time': self.execution_time
        }


class QueryOptimizer:
    """Optimizes queries for better performance."""
    
    def __init__(self, cache_query_plans: bool = True):
        """Initialize query optimizer.
        
        Args:
            cache_query_plans: Whether to cache query execution plans
        """
        self.cache_query_plans = cache_query_plans
        self._plan_cache: Dict[str, QueryPlan] = {}
        self._stats: Dict[str, Any] = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'optimizations_applied': 0
        }
    
    def optimize_query(self, query: Any, connection: Any) -> Tuple[Any, QueryPlan]:
        """Optimize a query and return execution plan.
        
        Args:
            query: SQLAlchemy query or raw SQL
            connection: Database connection
            
        Returns:
            Tuple of (optimized_query, query_plan)
        """
        self._stats['total_queries'] += 1
        
        # Get query hash for caching
        query_hash = self._get_query_hash(query)
        
        # Check cache
        if self.cache_query_plans and query_hash in self._plan_cache:
            self._stats['cache_hits'] += 1
            plan = self._plan_cache[query_hash]
            logger.debug(f"Query plan cache hit: {query_hash}")
            return query, plan
        
        self._stats['cache_misses'] += 1
        
        # Analyze query
        query_type = self._determine_query_type(query)
        
        # Generate execution plan
        plan = self._generate_execution_plan(query, query_type, connection)
        
        # Apply optimizations
        optimized_query = self._apply_optimizations(query, plan, connection)
        
        # Cache plan
        if self.cache_query_plans:
            self._plan_cache[query_hash] = plan
        
        return optimized_query, plan
    
    def _get_query_hash(self, query: Any) -> str:
        """Generate hash for query caching."""
        if isinstance(query, str):
            query_str = query
        elif hasattr(query, 'statement'):
            query_str = str(query.statement.compile())
        else:
            query_str = str(query)
        
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _determine_query_type(self, query: Any) -> QueryType:
        """Determine the type of query."""
        if isinstance(query, str):
            query_lower = query.lower().strip()
            if query_lower.startswith('select'):
                if 'group by' in query_lower or any(agg in query_lower for agg in ['count(', 'sum(', 'avg(', 'max(', 'min(']):
                    return QueryType.AGGREGATE
                elif 'join' in query_lower:
                    return QueryType.JOIN
                return QueryType.SELECT
            elif query_lower.startswith('insert'):
                return QueryType.INSERT
            elif query_lower.startswith('update'):
                return QueryType.UPDATE
            elif query_lower.startswith('delete'):
                return QueryType.DELETE
        
        # For SQLAlchemy queries
        if hasattr(query, 'statement'):
            # Simplified detection for SQLAlchemy
            return QueryType.SELECT
        
        return QueryType.SELECT
    
    def _generate_execution_plan(self, query: Any, query_type: QueryType, 
                                connection: Any) -> QueryPlan:
        """Generate execution plan for query."""
        plan = QueryPlan(
            query_type=query_type,
            estimated_cost=0.0,
            estimated_rows=0,
            uses_index=False
        )
        
        try:
            # For SQLite, use EXPLAIN QUERY PLAN
            if connection.dialect.name == 'sqlite':
                plan = self._analyze_sqlite_query(query, plan, connection)
            # For PostgreSQL, use EXPLAIN
            elif connection.dialect.name == 'postgresql':
                plan = self._analyze_postgresql_query(query, plan, connection)
            # For DuckDB, basic analysis
            elif connection.dialect.name == 'duckdb':
                plan = self._analyze_duckdb_query(query, plan, connection)
        except Exception as e:
            logger.warning(f"Could not analyze query plan: {e}")
        
        # Add optimization hints based on query type
        self._add_optimization_hints(plan)
        
        return plan
    
    def _analyze_sqlite_query(self, query: Any, plan: QueryPlan, 
                             connection: Any) -> QueryPlan:
        """Analyze SQLite query execution plan."""
        try:
            if isinstance(query, str):
                explain_query = f"EXPLAIN QUERY PLAN {query}"
            else:
                explain_query = f"EXPLAIN QUERY PLAN {str(query.statement.compile())}"
            
            result = connection.execute(text(explain_query))
            
            for row in result:
                detail = row[-1] if row else ""
                if "USING INDEX" in detail:
                    plan.uses_index = True
                    # Extract index name
                    import re
                    match = re.search(r'USING INDEX (\w+)', detail)
                    if match:
                        plan.index_names.append(match.group(1))
                
                if "SCAN TABLE" in detail:
                    # Full table scan detected
                    plan.optimization_hints.append("Consider adding index for better performance")
                    plan.estimated_cost += 100  # Arbitrary cost for table scan
                elif "SEARCH TABLE" in detail:
                    plan.estimated_cost += 10  # Lower cost for index search
            
        except Exception as e:
            logger.debug(f"SQLite query analysis failed: {e}")
        
        return plan
    
    def _analyze_postgresql_query(self, query: Any, plan: QueryPlan, 
                                 connection: Any) -> QueryPlan:
        """Analyze PostgreSQL query execution plan."""
        try:
            if isinstance(query, str):
                explain_query = f"EXPLAIN (FORMAT JSON) {query}"
            else:
                explain_query = f"EXPLAIN (FORMAT JSON) {str(query.statement.compile())}"
            
            result = connection.execute(text(explain_query))
            explain_data = result.scalar()
            
            if explain_data:
                plan_data = json.loads(explain_data)[0]['Plan']
                plan.estimated_cost = plan_data.get('Total Cost', 0)
                plan.estimated_rows = int(plan_data.get('Plan Rows', 0))
                
                # Check for index usage
                if 'Index' in plan_data.get('Node Type', ''):
                    plan.uses_index = True
                    if 'Index Name' in plan_data:
                        plan.index_names.append(plan_data['Index Name'])
                
        except Exception as e:
            logger.debug(f"PostgreSQL query analysis failed: {e}")
        
        return plan
    
    def _analyze_duckdb_query(self, query: Any, plan: QueryPlan, 
                             connection: Any) -> QueryPlan:
        """Analyze DuckDB query execution plan."""
        # DuckDB doesn't have traditional indexes like other databases
        # It uses zone maps and other optimization techniques
        plan.optimization_hints.append("DuckDB automatically optimizes queries")
        plan.uses_index = True  # DuckDB uses internal optimization
        
        return plan
    
    def _add_optimization_hints(self, plan: QueryPlan) -> None:
        """Add optimization hints based on query analysis."""
        if plan.query_type == QueryType.SELECT and not plan.uses_index:
            plan.optimization_hints.append("Consider adding index on frequently queried columns")
        
        if plan.query_type == QueryType.JOIN:
            plan.optimization_hints.append("Ensure join columns are indexed")
            plan.optimization_hints.append("Consider join order optimization")
        
        if plan.query_type == QueryType.AGGREGATE:
            plan.optimization_hints.append("Consider materialized views for complex aggregations")
            plan.optimization_hints.append("Ensure GROUP BY columns are indexed")
        
        if plan.estimated_rows > 10000:
            plan.optimization_hints.append("Consider pagination for large result sets")
            plan.optimization_hints.append("Use LIMIT clause when possible")
    
    def _apply_optimizations(self, query: Any, plan: QueryPlan, 
                           connection: Any) -> Any:
        """Apply query optimizations."""
        optimized_query = query
        
        # Apply different optimizations based on query type
        if plan.query_type == QueryType.SELECT:
            optimized_query = self._optimize_select_query(query, plan, connection)
        elif plan.query_type == QueryType.INSERT:
            optimized_query = self._optimize_insert_query(query, plan, connection)
        
        if optimized_query != query:
            self._stats['optimizations_applied'] += 1
            logger.debug("Query optimization applied")
        
        return optimized_query
    
    def _optimize_select_query(self, query: Any, plan: QueryPlan, 
                              connection: Any) -> Any:
        """Optimize SELECT queries."""
        # For SQLAlchemy queries, we can add query hints
        if hasattr(query, 'statement') and not plan.uses_index:
            # Log suggestion rather than modifying query
            logger.info("Query could benefit from index optimization")
        
        return query
    
    def _optimize_insert_query(self, query: Any, plan: QueryPlan, 
                              connection: Any) -> Any:
        """Optimize INSERT queries."""
        # Batch inserts are handled by BatchOptimizer
        return query
    
    def create_indexes(self, table_name: str, columns: List[str], 
                      connection: Any) -> List[str]:
        """Create indexes for specified columns.
        
        Args:
            table_name: Name of the table
            columns: List of column names to index
            connection: Database connection
            
        Returns:
            List of created index names
        """
        created_indexes = []
        
        try:
            inspector = inspect(connection)
            existing_indexes = inspector.get_indexes(table_name)
            existing_columns = {idx['column_names'][0] for idx in existing_indexes 
                              if len(idx['column_names']) == 1}
            
            for column in columns:
                if column not in existing_columns:
                    index_name = f"idx_{table_name}_{column}"
                    
                    # Create index
                    create_index_sql = f"CREATE INDEX {index_name} ON {table_name} ({column})"
                    connection.execute(text(create_index_sql))
                    
                    created_indexes.append(index_name)
                    logger.info(f"Created index: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
        
        return created_indexes
    
    def analyze_table_statistics(self, table_name: str, connection: Any) -> Dict[str, Any]:
        """Analyze table statistics for optimization.
        
        Args:
            table_name: Name of the table
            connection: Database connection
            
        Returns:
            Dictionary of table statistics
        """
        stats = {
            'row_count': 0,
            'indexes': [],
            'columns': [],
            'recommendations': []
        }
        
        try:
            # Get row count
            result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            stats['row_count'] = result.scalar()
            
            # Get table info
            inspector = inspect(connection)
            
            # Get columns
            columns = inspector.get_columns(table_name)
            stats['columns'] = [col['name'] for col in columns]
            
            # Get indexes
            indexes = inspector.get_indexes(table_name)
            stats['indexes'] = [{'name': idx['name'], 'columns': idx['column_names']} 
                               for idx in indexes]
            
            # Add recommendations
            if stats['row_count'] > 1000 and len(stats['indexes']) == 0:
                stats['recommendations'].append("Table has no indexes - consider adding for better query performance")
            
            # Check for common columns that should be indexed
            index_columns = {col for idx in stats['indexes'] for col in idx['columns']}
            for col in ['id', 'created_at', 'updated_at', 'name']:
                if col in stats['columns'] and col not in index_columns:
                    stats['recommendations'].append(f"Consider indexing column '{col}'")
            
        except Exception as e:
            logger.error(f"Failed to analyze table statistics: {e}")
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        cache_hit_rate = 0.0
        if self._stats['total_queries'] > 0:
            cache_hit_rate = self._stats['cache_hits'] / self._stats['total_queries']
        
        return {
            **self._stats,
            'cache_hit_rate': cache_hit_rate,
            'cached_plans': len(self._plan_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear query plan cache."""
        self._plan_cache.clear()
        logger.info("Query plan cache cleared")