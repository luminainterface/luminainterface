import logging
from central_node import CentralNode

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('processor_diagnostics.log')
        ]
    )
    return logging.getLogger('ProcessorDiagnostics')

def check_processor_health(central_node):
    issues = []
    for name, processor in central_node.processors.items():
        # Check initialization
        if not processor.is_initialized():
            issues.append(f"{name}: Not initialized")
            continue
            
        # Check activation
        if not processor.is_active():
            issues.append(f"{name}: Not active")
            continue
            
        # Check specific processor requirements
        if name == 'LanguageProcessor':
            if not processor.transformers_available:
                issues.append(f"{name}: Running in basic mode")
                
        elif name == 'HyperdimensionalThought':
            if not processor.memory:
                issues.append(f"{name}: Memory not initialized")
                
    return issues

def fix_processor_issues(central_node):
    logger = logging.getLogger('ProcessorFixer')
    
    # Step 1: Verify all required processors exist
    required_processors = [
        'NeuralProcessor',
        'LanguageProcessor',
        'HyperdimensionalThought'
    ]
    
    for proc_name in required_processors:
        if proc_name not in central_node.processors:
            logger.error(f"Missing processor: {proc_name}")
            continue
            
        processor = central_node.processors[proc_name]
        
        # Step 2: Fix initialization
        if not processor.is_initialized():
            logger.info(f"Initializing {proc_name}...")
            try:
                if not processor.initialize():
                    logger.error(f"Failed to initialize {proc_name}")
                    continue
            except Exception as e:
                logger.error(f"Error initializing {proc_name}: {str(e)}")
                continue
                
        # Step 3: Fix activation
        if not processor.is_active():
            logger.info(f"Activating {proc_name}...")
            try:
                if not processor.activate():
                    logger.error(f"Failed to activate {proc_name}")
                    continue
            except Exception as e:
                logger.error(f"Error activating {proc_name}: {str(e)}")
                continue
                
        # Step 4: Processor-specific fixes
        if proc_name == 'NeuralProcessor':
            # Ensure connection to language processor
            if not hasattr(processor, 'language_processor'):
                lang_proc = central_node.get_processor('LanguageProcessor')
                if lang_proc:
                    processor.connect_language_processor(lang_proc)
                    logger.info("Connected NeuralProcessor to LanguageProcessor")
                    
        elif proc_name == 'HyperdimensionalThought':
            # Ensure memory initialization
            if not processor.memory:
                processor._initialize_base_vectors()
                logger.info("Initialized HyperdimensionalThought memory")
                
    # Step 5: Verify fixes
    issues = check_processor_health(central_node)
    if issues:
        logger.warning("Remaining issues after fixes:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All processor issues resolved")
        
    return len(issues) == 0

def main():
    logger = setup_logging()
    
    try:
        # Initialize central node
        logger.info("Initializing CentralNode...")
        central_node = CentralNode()
        
        # Check initial state
        logger.info("Checking initial processor state...")
        initial_issues = check_processor_health(central_node)
        if initial_issues:
            logger.info("Found processor issues:")
            for issue in initial_issues:
                logger.info(f"  - {issue}")
                
            # Attempt to fix issues
            logger.info("Attempting to fix processor issues...")
            success = fix_processor_issues(central_node)
            
            if success:
                logger.info("Successfully fixed all processor issues")
            else:
                logger.error("Some processor issues remain")
        else:
            logger.info("No processor issues found")
            
    except Exception as e:
        logger.error(f"Error during processor diagnostics: {str(e)}")
        return False
        
    return True

if __name__ == "__main__":
    main() 
 
 
from central_node import CentralNode

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('processor_diagnostics.log')
        ]
    )
    return logging.getLogger('ProcessorDiagnostics')

def check_processor_health(central_node):
    issues = []
    for name, processor in central_node.processors.items():
        # Check initialization
        if not processor.is_initialized():
            issues.append(f"{name}: Not initialized")
            continue
            
        # Check activation
        if not processor.is_active():
            issues.append(f"{name}: Not active")
            continue
            
        # Check specific processor requirements
        if name == 'LanguageProcessor':
            if not processor.transformers_available:
                issues.append(f"{name}: Running in basic mode")
                
        elif name == 'HyperdimensionalThought':
            if not processor.memory:
                issues.append(f"{name}: Memory not initialized")
                
    return issues

def fix_processor_issues(central_node):
    logger = logging.getLogger('ProcessorFixer')
    
    # Step 1: Verify all required processors exist
    required_processors = [
        'NeuralProcessor',
        'LanguageProcessor',
        'HyperdimensionalThought'
    ]
    
    for proc_name in required_processors:
        if proc_name not in central_node.processors:
            logger.error(f"Missing processor: {proc_name}")
            continue
            
        processor = central_node.processors[proc_name]
        
        # Step 2: Fix initialization
        if not processor.is_initialized():
            logger.info(f"Initializing {proc_name}...")
            try:
                if not processor.initialize():
                    logger.error(f"Failed to initialize {proc_name}")
                    continue
            except Exception as e:
                logger.error(f"Error initializing {proc_name}: {str(e)}")
                continue
                
        # Step 3: Fix activation
        if not processor.is_active():
            logger.info(f"Activating {proc_name}...")
            try:
                if not processor.activate():
                    logger.error(f"Failed to activate {proc_name}")
                    continue
            except Exception as e:
                logger.error(f"Error activating {proc_name}: {str(e)}")
                continue
                
        # Step 4: Processor-specific fixes
        if proc_name == 'NeuralProcessor':
            # Ensure connection to language processor
            if not hasattr(processor, 'language_processor'):
                lang_proc = central_node.get_processor('LanguageProcessor')
                if lang_proc:
                    processor.connect_language_processor(lang_proc)
                    logger.info("Connected NeuralProcessor to LanguageProcessor")
                    
        elif proc_name == 'HyperdimensionalThought':
            # Ensure memory initialization
            if not processor.memory:
                processor._initialize_base_vectors()
                logger.info("Initialized HyperdimensionalThought memory")
                
    # Step 5: Verify fixes
    issues = check_processor_health(central_node)
    if issues:
        logger.warning("Remaining issues after fixes:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All processor issues resolved")
        
    return len(issues) == 0

def main():
    logger = setup_logging()
    
    try:
        # Initialize central node
        logger.info("Initializing CentralNode...")
        central_node = CentralNode()
        
        # Check initial state
        logger.info("Checking initial processor state...")
        initial_issues = check_processor_health(central_node)
        if initial_issues:
            logger.info("Found processor issues:")
            for issue in initial_issues:
                logger.info(f"  - {issue}")
                
            # Attempt to fix issues
            logger.info("Attempting to fix processor issues...")
            success = fix_processor_issues(central_node)
            
            if success:
                logger.info("Successfully fixed all processor issues")
            else:
                logger.error("Some processor issues remain")
        else:
            logger.info("No processor issues found")
            
    except Exception as e:
        logger.error(f"Error during processor diagnostics: {str(e)}")
        return False
        
    return True

if __name__ == "__main__":
    main() 
 