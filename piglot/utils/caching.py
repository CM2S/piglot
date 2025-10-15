"""Module for caching utilities using numpy arrays."""
from typing import Callable, List, Tuple, TypeVar, Optional, Generic
from dataclasses import dataclass
from threading import RLock, Condition
import numpy as np

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Represents a single entry in the cache."""
    entry: np.ndarray
    value: T = None


class LRUCache(Generic[T]):
    """A thread-safe LRU (Least Recently Used) cache implementation."""

    def __init__(self, capacity: int, compute_func: Callable[[np.ndarray], T]) -> None:
        self.capacity = capacity
        self.cache: List[Optional[CacheEntry[T]]] = [None for _ in range(capacity)]
        self.compute_func = compute_func
        self.mutex = RLock()
        self.cvar = Condition(lock=self.mutex)
        self.stats = {
            "hits": 0,
            "misses": 0,
        }

    def __probe(self, entry: np.ndarray) -> Tuple[Optional[int], Optional[CacheEntry]]:
        """Probe the cache for a given entry without modifying the cache state.

        Parameters
        ----------
        entry : np.ndarray
            The input array to look up in the cache.

        Returns
        -------
        Tuple[Optional[int], Optional[CacheEntry]]
            The index and cache entry if found, otherwise None.
        """
        for i, cached_entry in enumerate(self.cache):
            if cached_entry is not None and np.array_equal(cached_entry.entry, entry):
                return i, cached_entry
        return None, None

    def __update(self, pos: int, value: Optional[T] = None) -> None:
        """Update the value of a cache entry at a given position and move it to the front.

        Parameters
        ----------
        pos : int
            The current position of the entry in the cache.
        """
        if value is not None:
            self.cache[pos].value = value
        self.cache.insert(0, self.cache.pop(pos))

    def __push(self, entry: CacheEntry[T]) -> None:
        """Push new an entry to the front of the cache, evicting the least recently used one.

        Parameters
        ----------
        entry : CacheEntry
            The cache entry to push to the front.
        """
        self.cache.insert(0, entry)
        self.cache.pop()

    def get(self, entry: np.ndarray) -> T:
        """Get the value from an item. If not cached, compute it using the provided function.

        Parameters
        ----------
        entry : np.ndarray
            The input array to look up in the cache.

        Returns
        -------
        T
            The cached or computed value.
        """
        # Probe cache status
        with self.mutex:
            i, probe_result = self.__probe(entry)
            entry_in_cache = probe_result is not None
            if entry_in_cache:
                # Move the accessed entry to the front (most recently used)
                self.__update(i)

                # Cache hit: return the value if computed
                if probe_result.value is not None:
                    self.stats["hits"] += 1
                    return probe_result.value
            else:
                # Not found in cache: create a new entry to indicate computation in progress
                self.__push(CacheEntry(entry=entry.copy()))

        # If someone else is working on this entry, wait for them to finish
        if entry_in_cache:
            while True:
                with self.mutex:
                    _, probe_result = self.__probe(entry)

                # In rare cases, the entry might have been evicted: exit and recompute result
                if probe_result is None:
                    break

                # Check if the value is now available
                if probe_result.value is not None:
                    self.stats["hits"] += 1
                    return probe_result.value

                # Wait for someone to notify us that a new value is available
                with self.cvar:
                    self.cvar.wait()

        # Compute the value outside the lock
        computed_value = self.compute_func(entry)

        # Update the cache with the computed value
        with self.mutex:
            i, probe_result = self.__probe(entry)
            # Update the cache: if the entry was evicted, create a new one
            if probe_result is None:
                self.__push(CacheEntry(entry=entry.copy(), value=computed_value))
            else:
                self.__update(i, value=computed_value)

        # Notify any waiting threads that a new value is available
        with self.cvar:
            self.cvar.notify_all()

        self.stats["misses"] += 1
        return computed_value
