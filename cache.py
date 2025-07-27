# cache.py
import hashlib
import json
from datetime import datetime, timedelta
import pickle
from pathlib import Path  # This is what Path refers to

class QueryCache:
    def __init__(self, cache_dir="cache", ttl_hours=24):
        self.cache_dir = Path(cache_dir)  # Creates a Path object
        self.cache_dir.mkdir(exist_ok=True)  # Creates directory if it doesn't exist
        self.ttl = timedelta(hours=ttl_hours)
        
    def _get_cache_key(self, query, k=5):
        """Generate cache key from query"""
        key_str = f"{query}_{k}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query, k=5):
        """Get cached result if exists and not expired"""
        cache_key = self._get_cache_key(query, k)
        cache_file = self.cache_dir / f"{cache_key}.pkl"  # Path concatenation
        
        if cache_file.exists():  # Check if file exists
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if expired
            if datetime.now() - cached_data['timestamp'] < self.ttl:
                return cached_data['result']
        
        return None
    
    def set(self, query, result, k=5):
        """Cache the result"""
        cache_key = self._get_cache_key(query, k)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        cached_data = {
            'query': query,
            'result': result,
            'timestamp': datetime.now(),
            'k': k
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
    
    def clear_expired(self):
        """Remove expired cache files"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if datetime.now() - cached_data['timestamp'] >= self.ttl:
                    cache_file.unlink()  # Delete the file
            except:
                # If we can't read it, delete it
                cache_file.unlink()