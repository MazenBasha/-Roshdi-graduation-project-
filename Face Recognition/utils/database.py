"""
Database utilities for storing and managing face templates.

Provides JSON-based storage for face embeddings and metadata.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


class TemplateDatabase:
    """Manages face templates in JSON format."""

    def __init__(self, db_path: Path):
        """
        Initialize template database.

        Args:
            db_path: Path to JSON database file.
        """
        self.db_path = Path(db_path)

        # Create directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing database or create new one
        self.templates = self._load_db()

    def _load_db(self) -> dict:
        """Load existing database from JSON file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load database from {self.db_path}")
                return {"templates": {}}
        return {"templates": {}}

    def _save_db(self) -> None:
        """Save database to JSON file."""
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.templates, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to save database: {str(e)}")

    def add_template(
        self,
        person_name: str,
        embedding: np.ndarray,
        num_samples: int = 1,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add or update a face template.

        Args:
            person_name: Name of the person.
            embedding: Face embedding vector (should be L2-normalized).
            num_samples: Number of images used to compute embedding.
            metadata: Optional metadata dictionary.
        """
        # Ensure embedding is 1D
        embedding = embedding.flatten()

        # Create template entry
        template = {
            "name": person_name,
            "embedding": embedding.tolist(),
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
        }

        if metadata:
            template["metadata"] = metadata

        # Store in database
        if "templates" not in self.templates:
            self.templates["templates"] = {}

        self.templates["templates"][person_name] = template

        # Save to file
        self._save_db()

        print(f"Added template for '{person_name}' ({num_samples} samples)")

    def get_template(self, person_name: str) -> Optional[np.ndarray]:
        """
        Retrieve a template embedding by name.

        Args:
            person_name: Name of the person.

        Returns:
            Embedding as numpy array, or None if not found.
        """
        if "templates" not in self.templates:
            return None

        template = self.templates["templates"].get(person_name)

        if template is None:
            return None

        return np.array(template["embedding"])

    def get_all_templates(self) -> Dict[str, np.ndarray]:
        """
        Get all templates.

        Returns:
            Dictionary mapping person_name -> embedding.
        """
        templates = {}

        if "templates" not in self.templates:
            return templates

        for person_name, template in self.templates["templates"].items():
            templates[person_name] = np.array(template["embedding"])

        return templates

    def get_template_metadata(self, person_name: str) -> Optional[dict]:
        """
        Get metadata for a template.

        Args:
            person_name: Name of the person.

        Returns:
            Metadata dictionary, or None if not found.
        """
        if "templates" not in self.templates:
            return None

        template = self.templates["templates"].get(person_name)

        if template is None:
            return None

        return {
            "name": template.get("name"),
            "num_samples": template.get("num_samples"),
            "timestamp": template.get("timestamp"),
            "metadata": template.get("metadata", {}),
        }

    def remove_template(self, person_name: str) -> bool:
        """
        Remove a template.

        Args:
            person_name: Name of the person.

        Returns:
            True if removed, False if not found.
        """
        if "templates" not in self.templates:
            return False

        if person_name in self.templates["templates"]:
            del self.templates["templates"][person_name]
            self._save_db()
            print(f"Removed template for '{person_name}'")
            return True

        return False

    def list_all_persons(self) -> List[str]:
        """
        Get list of all enrolled persons.

        Returns:
            List of person names.
        """
        if "templates" not in self.templates:
            return []

        return list(self.templates["templates"].keys())

    def clear_database(self) -> None:
        """Clear all templates from database."""
        self.templates = {"templates": {}}
        self._save_db()
        print("Database cleared")

    def size(self) -> int:
        """Get number of templates in database."""
        if "templates" not in self.templates:
            return 0

        return len(self.templates["templates"])

    def export_json(self) -> str:
        """
        Export database as JSON string.

        Returns:
            JSON string representation of database.
        """
        return json.dumps(self.templates, indent=2)

    def import_json(self, json_str: str) -> None:
        """
        Import database from JSON string.

        Args:
            json_str: JSON string containing templates.
        """
        try:
            data = json.loads(json_str)
            self.templates = data
            self._save_db()
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {str(e)}")


def create_database_if_needed(db_path: Path) -> None:
    """
    Create an empty database file if it doesn't exist.

    Args:
        db_path: Path to database file.
    """
    if not db_path.exists():
        db = TemplateDatabase(db_path)
        print(f"Created new database at {db_path}")
