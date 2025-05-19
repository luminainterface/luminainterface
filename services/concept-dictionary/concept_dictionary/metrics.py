from prometheus_client import Counter, Histogram, Gauge

# Operation counters
CONCEPT_OPERATIONS = Counter(
    'concept_dictionary_operations_total',
    'Total number of concept operations',
    ['operation']  # 'create', 'update', 'delete', etc.
)

CONCEPT_SYNC_OPERATIONS = Counter(
    'concept_dictionary_sync_operations_total',
    'Total number of concept sync operations',
    ['operation']  # 'redis_to_qdrant', 'qdrant_to_redis', etc.
)

CONCEPT_DIGEST_OPERATIONS = Counter(
    'concept_dictionary_digest_operations_total',
    'Total number of concept digest operations',
    ['operation']  # 'digest', 'quality_check', 'merge', etc.
)

# Latency histograms
CONCEPT_OPERATION_LATENCY = Histogram(
    'concept_dictionary_operation_latency_seconds',
    'Time spent on concept operations',
    ['operation']  # 'create', 'update', 'delete', etc.
)

CONCEPT_SYNC_LATENCY = Histogram(
    'concept_dictionary_sync_latency_seconds',
    'Time spent on concept sync operations',
    ['operation']  # 'redis_to_qdrant', 'qdrant_to_redis', etc.
)

CONCEPT_DIGEST_LATENCY = Histogram(
    'concept_dictionary_digest_latency_seconds',
    'Time spent on concept digest operations',
    ['operation']  # 'digest', 'quality_check', 'merge', etc.
)

# Quality metrics
CONCEPT_QUALITY_SCORE = Gauge(
    'concept_dictionary_quality_score',
    'Quality score of concepts',
    ['term']  # The concept term
)

# Error counters
CONCEPT_SYNC_ERRORS = Counter(
    'concept_dictionary_sync_errors_total',
    'Total number of concept sync errors',
    ['error_type']  # 'redis_error', 'qdrant_error', etc.
)

CONCEPT_DIGEST_ERRORS = Counter(
    'concept_dictionary_digest_errors_total',
    'Total number of concept digest errors',
    ['error_type']  # 'digest_error', 'quality_check_error', etc.
)

# Queue size gauges
CONCEPT_QUEUE_SIZE = Gauge(
    'concept_dictionary_queue_size',
    'Current size of the concept queue',
    ['queue_type']  # 'sync', 'digest', 'training', etc.
)

CONCEPT_DIGEST_QUEUE_SIZE = Gauge(
    'concept_dictionary_digest_queue_size',
    'Current size of the concept digest queue'
)

CONCEPT_TRAINING_QUEUE_SIZE = Gauge(
    'concept_dictionary_training_queue_size',
    'Current size of the concept training queue'
)

# Training metrics
CONCEPT_TRAINING_STATUS = Gauge(
    'concept_dictionary_training_status',
    'Training status of concepts',
    ['term', 'status']  # The concept term and status ('pending', 'training', 'trained', 'failed')
)

CONCEPT_TRAINING_LATENCY = Histogram(
    'concept_dictionary_training_latency_seconds',
    'Time spent training concepts',
    ['term']  # The concept term
)

CONCEPT_TRAINING_ERRORS = Counter(
    'concept_dictionary_training_errors_total',
    'Total number of concept training errors',
    ['term', 'error_type']  # The concept term and error type
) 