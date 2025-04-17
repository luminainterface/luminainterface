import argparse
import logging
from central_node import CentralNode

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Central Node CLI')
    parser.add_argument('--model', type=str, default="mistralai/Mistral-7B-v0.1",
                      help='Path to Mistral model')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize central node
        node = CentralNode(args.model)
        logger.info("Central Node initialized")

        # Start conversation
        conv_id = node.start_conversation()
        logger.info(f"Started conversation {conv_id}")

        print("\nWelcome to the Central Node CLI!")
        print("Type 'exit' to end the conversation")
        print("Type 'history' to view conversation history")
        print("Type 'state' to view current neural state")
        print("Type 'save' to save the current state")
        print("Type 'load' to load previous state")
        print("\nStart chatting:")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                node.end_conversation(conv_id)
                logger.info("Conversation ended")
                break

            elif user_input.lower() == 'history':
                history = node.get_conversation_history(conv_id)
                print("\nConversation History:")
                for msg in history:
                    print(f"\nUser: {msg['user_message']}")
                    print(f"System: {msg['system_response']}")
                    print(f"State: {msg['neural_state']}")

            elif user_input.lower() == 'state':
                state = node.neural_network.get_neural_state()
                print("\nCurrent Neural State:")
                print(f"Temperature: {state['temperature']}")
                print(f"Top P: {state['top_p']}")
                print(f"LLM Weight: {state['llm_weight']}")
                print(f"Neural Weight: {state['neural_weight']}")

            elif user_input.lower() == 'save':
                node.save_state()
                print("\nState saved successfully")

            elif user_input.lower() == 'load':
                node.load_state()
                print("\nState loaded successfully")

            else:
                # Process message
                response = node.process_message(conv_id, user_input)
                print(f"\nSystem: {response['response']}")
                print(f"Neural State: {response['neural_state']}")

    except Exception as e:
        logger.error(f"Error in CLI: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
 
 
import logging
from central_node import CentralNode

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description='Central Node CLI')
    parser.add_argument('--model', type=str, default="mistralai/Mistral-7B-v0.1",
                      help='Path to Mistral model')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize central node
        node = CentralNode(args.model)
        logger.info("Central Node initialized")

        # Start conversation
        conv_id = node.start_conversation()
        logger.info(f"Started conversation {conv_id}")

        print("\nWelcome to the Central Node CLI!")
        print("Type 'exit' to end the conversation")
        print("Type 'history' to view conversation history")
        print("Type 'state' to view current neural state")
        print("Type 'save' to save the current state")
        print("Type 'load' to load previous state")
        print("\nStart chatting:")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                node.end_conversation(conv_id)
                logger.info("Conversation ended")
                break

            elif user_input.lower() == 'history':
                history = node.get_conversation_history(conv_id)
                print("\nConversation History:")
                for msg in history:
                    print(f"\nUser: {msg['user_message']}")
                    print(f"System: {msg['system_response']}")
                    print(f"State: {msg['neural_state']}")

            elif user_input.lower() == 'state':
                state = node.neural_network.get_neural_state()
                print("\nCurrent Neural State:")
                print(f"Temperature: {state['temperature']}")
                print(f"Top P: {state['top_p']}")
                print(f"LLM Weight: {state['llm_weight']}")
                print(f"Neural Weight: {state['neural_weight']}")

            elif user_input.lower() == 'save':
                node.save_state()
                print("\nState saved successfully")

            elif user_input.lower() == 'load':
                node.load_state()
                print("\nState loaded successfully")

            else:
                # Process message
                response = node.process_message(conv_id, user_input)
                print(f"\nSystem: {response['response']}")
                print(f"Neural State: {response['neural_state']}")

    except Exception as e:
        logger.error(f"Error in CLI: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
 