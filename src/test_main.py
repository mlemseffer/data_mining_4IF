"""
Script de test pour main.py - Lance automatiquement l'option 5
"""
import sys
from io import StringIO

# Simuler les entrées utilisateur
sys.stdin = StringIO("5\n10\n")

# Importer et exécuter main
from main import main

if __name__ == '__main__':
    main()
