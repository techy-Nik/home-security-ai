"""Structured logging with rotation"""
import logging
import json
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any


class StructuredLogger:
    """Logger with structured output for analysis"""
    
    def __init__(self, config):
        self.logger = logging.getLogger('SecuritySystem')
        self.logger.setLevel(getattr(logging, config.logging.level))
        
        # Console handler with UTF-8 encoding fix for Windows
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Force UTF-8 encoding for Windows console
        if sys.platform == 'win32':
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation (UTF-8 encoding)
        log_file = Path(config.logging.file)
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding='utf-8'  # Explicit UTF-8 for file
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Structured event log (JSON format for analysis)
        self.event_log_file = log_file.parent / "events.jsonl"
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log structured event to JSONL file"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **data
        }
        
        try:
            with open(self.event_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write event log: {e}")
    
    def info(self, msg: str):
        # Replace problematic Unicode characters on Windows
        if sys.platform == 'win32':
            msg = msg.replace('✓', '[OK]').replace('🎥', '[CAM]').replace('🚨', '[ALERT]')
            msg = msg.replace('⚠️', '[WARN]').replace('📸', '[SNAP]').replace('💾', '[SAVE]')
            msg = msg.replace('⏸', '[PAUSE]').replace('▶', '[PLAY]').replace('📊', '[STATS]')
            msg = msg.replace('👁️', '[WATCH]').replace('ℹ️', '[INFO]').replace('📍', '[LOC]')
        self.logger.info(msg)
    
    def warning(self, msg: str):
        if sys.platform == 'win32':
            msg = msg.replace('✓', '[OK]').replace('🎥', '[CAM]').replace('🚨', '[ALERT]')
            msg = msg.replace('⚠️', '[WARN]').replace('📸', '[SNAP]').replace('💾', '[SAVE]')
            msg = msg.replace('⏸', '[PAUSE]').replace('▶', '[PLAY]').replace('📊', '[STATS]')
            msg = msg.replace('👁️', '[WATCH]').replace('ℹ️', '[INFO]').replace('📍', '[LOC]')
        self.logger.warning(msg)
    
    def error(self, msg: str, exc_info=False):
        if sys.platform == 'win32':
            msg = msg.replace('✓', '[OK]').replace('🎥', '[CAM]').replace('🚨', '[ALERT]')
            msg = msg.replace('⚠️', '[WARN]').replace('📸', '[SNAP]').replace('💾', '[SAVE]')
            msg = msg.replace('⏸', '[PAUSE]').replace('▶', '[PLAY]').replace('📊', '[STATS]')
            msg = msg.replace('👁️', '[WATCH]').replace('ℹ️', '[INFO]').replace('📍', '[LOC]')
        self.logger.error(msg, exc_info=exc_info)
    
    def debug(self, msg: str):
        if sys.platform == 'win32':
            msg = msg.replace('✓', '[OK]').replace('🎥', '[CAM]').replace('🚨', '[ALERT]')
            msg = msg.replace('⚠️', '[WARN]').replace('📸', '[SNAP]').replace('💾', '[SAVE]')
            msg = msg.replace('⏸', '[PAUSE]').replace('▶', '[PLAY]').replace('📊', '[STATS]')
            msg = msg.replace('👁️', '[WATCH]').replace('ℹ️', '[INFO]').replace('📍', '[LOC]')
        self.logger.debug(msg)