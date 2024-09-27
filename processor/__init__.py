from processor.train_processor import train_processor
from processor.valid_processor import valid_processor
from processor.test_processor import test_processor

PROCESSOR_CLASSES = {
    'train_processor': train_processor,
    'valid_processor': valid_processor,
    'test_processor': test_processor,
}